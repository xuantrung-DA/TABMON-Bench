"""Faithful and auditable Explanatory Performance Estimation (XPE).

The implementation follows Decker et al. (KDD 2024) and their MIT-licensed
reference implementation (``thomdeck/xpe``, commit
``bac9ef37d8409b5eebf31b363d14864c763375c5``): EMD transport, the
maximum-coupling source match, its source label as anticipated target label,
and Shapley allocation of the matched log-loss change.

Source and target transport samples are deliberately equal-sized. This is a
load-bearing fidelity condition: with uniform masses, the EMD solution is
one-to-one (up to ties), so ``argmax`` does not discard most of a target
column's coupling mass. Two attribution backends are available:

* ``kernel_shap`` mirrors the authors' implementation and defaults to their
  reported 3,000 SHAP samples;
* ``grouped_permutation`` is an unbiased, batched Shapley estimator for the
  scalable full grid. Its efficiency residual and agreement with KernelSHAP
  must be reported before it is used as an XPE result in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.risk import multiclass_log_losses


METHOD = "xpe"
ENDPOINT = "transport_anticipated_log_loss_change"
OFFICIAL_REPOSITORY = "https://github.com/thomdeck/xpe"
OFFICIAL_COMMIT = "bac9ef37d8409b5eebf31b363d14864c763375c5"
IMPLEMENTATION_VERSION = "schema13_numeric_domain_restore_v3"
SUPPORTED_BACKENDS = {"grouped_permutation", "kernel_shap"}
BACKEND_METHOD_NAMES = {
    "grouped_permutation": "xpe_grouped_permutation",
    "kernel_shap": "xpe_reference_kernel_shap",
}


def _transport_representation(
    source: pd.DataFrame, target: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic numeric representation for EMD only."""
    combined = pd.concat([source, target], ignore_index=True)
    categorical = [
        column
        for column in combined.columns
        if not pd.api.types.is_numeric_dtype(combined[column])
    ]
    encoded = pd.get_dummies(
        combined, columns=categorical, dummy_na=True, dtype=float
    ).astype(float)
    values = encoded.to_numpy(dtype=np.float64)
    mean = values[: len(source)].mean(axis=0)
    std = values[: len(source)].std(axis=0)
    std[std < 1e-12] = 1.0
    values = (values - mean) / std
    return values[: len(source)], values[len(source) :]


def _loss_for_labels(model: Any, rows: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(rows), dtype=float)
    return multiclass_log_losses(probabilities, np.asarray(model.classes_), labels)


def _restore_frame(array: np.ndarray, template: pd.DataFrame) -> pd.DataFrame:
    """Restore a model-facing frame without narrowing shifted numeric values.

    KernelSHAP constructs coalitions as NumPy arrays.  Mixed tabular frames
    therefore lose their column dtypes.  Numeric columns must not be cast back
    to a narrow source dtype: a pipeline sentinel such as ``-999`` promotes an
    ``int8`` source feature in the target, and casting the coalition back to
    ``int8`` silently wraps the sentinel.  Conversely, an unconditional cast
    to float64 can change a float32 preprocessing path at a downstream tree
    threshold.  We therefore preserve floating dtypes and promote integer
    dtypes only as far as required by the realized coalition values.
    """
    frame = pd.DataFrame(array, columns=template.columns)
    for column in template.columns:
        dtype = template[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if pd.api.types.is_float_dtype(dtype):
                frame[column] = numeric.astype(dtype)
            elif pd.api.types.is_integer_dtype(dtype) and not numeric.isna().any():
                safe_dtype = np.dtype(dtype)
                safe_dtype = np.promote_types(
                    safe_dtype, np.min_scalar_type(int(numeric.min()))
                )
                safe_dtype = np.promote_types(
                    safe_dtype, np.min_scalar_type(int(numeric.max()))
                )
                frame[column] = numeric.astype(safe_dtype)
            else:
                frame[column] = numeric.astype(np.float64)
        else:
            frame[column] = frame[column].astype(object)
    return frame


@dataclass(frozen=True)
class XPEBatchResult:
    method: str
    attribution: dict[str, float]
    estimated_loss_change: float
    matched_target_loss: float
    matched_source_loss: float
    explained_targets: int
    source_transport_rows: int
    target_transport_rows: int
    coupling_retained_mass_fraction: float
    coupling_one_to_one_fraction: float
    shapley_efficiency_max_abs_error: float
    attribution_backend: str
    attribution_budget: int
    target_sample_positions: np.ndarray
    matched_source_positions: np.ndarray
    anticipated_labels: np.ndarray
    per_target_loss_change: np.ndarray
    per_target_attribution: np.ndarray


class XPEExplainer:
    """Source-fitted XPE explainer that never accepts target labels."""

    def __init__(self, source_X: pd.DataFrame, source_y: pd.Series):
        if len(source_X) != len(source_y):
            raise ValueError("Source features and labels are not aligned")
        if len(source_X) < 2:
            raise ValueError("XPE requires at least two source rows")
        self.source_X = source_X.reset_index(drop=True).copy()
        self.source_y = source_y.reset_index(drop=True).copy()

    @staticmethod
    def _grouped_permutation_values(
        model: Any,
        source: pd.DataFrame,
        source_y: pd.Series,
        target: pd.DataFrame,
        matched_source: np.ndarray,
        permutations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        features = list(source.columns)
        rows: list[pd.Series] = []
        labels: list[Any] = []
        orders: list[tuple[int, np.ndarray]] = []
        for target_position, source_position in enumerate(matched_source):
            baseline = source.iloc[source_position].copy()
            target_row = target.iloc[target_position]
            anticipated_label = source_y.iloc[source_position]
            orders_for_target: list[np.ndarray] = []
            for _ in range(permutations // 2):
                order = rng.permutation(len(features))
                orders_for_target.extend([order, order[::-1]])
            if permutations % 2:
                orders_for_target.append(rng.permutation(len(features)))
            for order in orders_for_target:
                working = baseline.copy()
                rows.append(working.copy())
                for feature_index in order:
                    feature = features[feature_index]
                    working = working.copy()
                    working[feature] = target_row[feature]
                    rows.append(working.copy())
                labels.extend([anticipated_label] * (len(features) + 1))
                orders.append((target_position, order))

        losses = _loss_for_labels(
            model,
            pd.DataFrame(rows, columns=features),
            np.asarray(labels),
        )
        values = np.zeros((len(target), len(features)), dtype=np.float64)
        width = len(features) + 1
        offset = 0
        for target_position, order in orders:
            values[target_position, order] += np.diff(losses[offset : offset + width])
            offset += width
        values /= float(permutations)
        return values

    @staticmethod
    def _kernel_shap_values(
        model: Any,
        source: pd.DataFrame,
        source_y: pd.Series,
        target: pd.DataFrame,
        matched_source: np.ndarray,
        nsamples: int,
        seed: int,
    ) -> np.ndarray:
        try:
            import shap
        except ImportError as exc:  # pragma: no cover - environment validation
            raise ImportError("KernelSHAP XPE requires the 'shap' package") from exc

        values = np.zeros((len(target), source.shape[1]), dtype=np.float64)
        # KernelExplainer uses NumPy's legacy global RNG internally.
        previous_state = np.random.get_state()
        np.random.seed(seed)
        try:
            for target_position, source_position in enumerate(matched_source):
                baseline = source.iloc[[source_position]].copy()
                target_row = target.iloc[[target_position]].copy()
                anticipated_label = source_y.iloc[source_position]

                def forward(array: np.ndarray) -> np.ndarray:
                    frame = _restore_frame(np.asarray(array), source)
                    labels = np.full(len(frame), anticipated_label)
                    return _loss_for_labels(model, frame, labels)

                explainer = shap.KernelExplainer(forward, baseline.to_numpy())
                explained = explainer.shap_values(
                    target_row.to_numpy(), nsamples=nsamples, silent=True
                )
                array = np.asarray(explained, dtype=np.float64)
                values[target_position] = array.reshape(-1, source.shape[1])[0]
        finally:
            np.random.set_state(previous_state)
        return values

    def explain(
        self,
        model: Any,
        target_X: pd.DataFrame,
        *,
        sample_size: int = 100,
        attribution_backend: str = "grouped_permutation",
        permutations: int = 128,
        kernel_nsamples: int = 3000,
        random_seed: int = 42,
    ) -> XPEBatchResult:
        if list(target_X.columns) != list(self.source_X.columns):
            raise ValueError("Source and target feature schemas differ")
        if sample_size < 2:
            raise ValueError("sample_size must be at least two")
        if attribution_backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported XPE backend: {attribution_backend}")
        if permutations < 1 or kernel_nsamples < 1:
            raise ValueError("XPE attribution budgets must be positive")

        rng = np.random.default_rng(random_seed)
        common_n = min(sample_size, len(self.source_X), len(target_X))
        if common_n < 2:
            raise ValueError("XPE requires at least two rows in each domain")
        source_idx = rng.choice(len(self.source_X), common_n, replace=False)
        target_idx = rng.choice(len(target_X), common_n, replace=False)
        source = self.source_X.iloc[source_idx].reset_index(drop=True)
        source_y = self.source_y.iloc[source_idx].reset_index(drop=True)
        target = target_X.iloc[target_idx].reset_index(drop=True)

        try:
            import ot
        except ImportError as exc:  # pragma: no cover - environment validation
            raise ImportError("XPE requires the 'pot' package") from exc
        source_repr, target_repr = _transport_representation(source, target)
        transport = ot.da.EMDTransport()
        transport.fit(Xs=source_repr, Xt=target_repr)
        coupling = np.asarray(transport.coupling_, dtype=np.float64)
        expected_mass = np.full(common_n, 1.0 / common_n)
        if not np.allclose(coupling.sum(axis=0), expected_mass, atol=1e-7):
            raise RuntimeError("XPE coupling does not preserve target mass")
        if not np.allclose(coupling.sum(axis=1), expected_mass, atol=1e-7):
            raise RuntimeError("XPE coupling does not preserve source mass")
        matched_source = np.argmax(coupling, axis=0)
        retained = coupling[matched_source, np.arange(common_n)] / expected_mass
        one_to_one = len(np.unique(matched_source)) / float(common_n)

        anticipated_labels = source_y.iloc[matched_source].to_numpy()
        target_losses = _loss_for_labels(model, target, anticipated_labels)
        matched_source_frame = source.iloc[matched_source].reset_index(drop=True)
        source_losses = _loss_for_labels(
            model, matched_source_frame, anticipated_labels
        )
        per_target_change = target_losses - source_losses

        if attribution_backend == "kernel_shap":
            per_target_attribution = self._kernel_shap_values(
                model,
                source,
                source_y,
                target,
                matched_source,
                kernel_nsamples,
                random_seed,
            )
            budget = kernel_nsamples
        else:
            per_target_attribution = self._grouped_permutation_values(
                model,
                source,
                source_y,
                target,
                matched_source,
                permutations,
                rng,
            )
            budget = permutations

        efficiency_error = np.abs(
            per_target_attribution.sum(axis=1) - per_target_change
        )
        mean_attribution = per_target_attribution.mean(axis=0)
        mean_target = float(np.mean(target_losses))
        mean_source = float(np.mean(source_losses))
        return XPEBatchResult(
            method=BACKEND_METHOD_NAMES[attribution_backend],
            attribution={
                feature: float(value)
                for feature, value in zip(source.columns, mean_attribution)
            },
            estimated_loss_change=mean_target - mean_source,
            matched_target_loss=mean_target,
            matched_source_loss=mean_source,
            explained_targets=common_n,
            source_transport_rows=common_n,
            target_transport_rows=common_n,
            coupling_retained_mass_fraction=float(np.mean(retained)),
            coupling_one_to_one_fraction=float(one_to_one),
            shapley_efficiency_max_abs_error=float(np.max(efficiency_error)),
            attribution_backend=attribution_backend,
            attribution_budget=int(budget),
            target_sample_positions=target_idx.astype(np.int64),
            matched_source_positions=source_idx[matched_source].astype(np.int64),
            anticipated_labels=anticipated_labels,
            per_target_loss_change=per_target_change,
            per_target_attribution=per_target_attribution,
        )
