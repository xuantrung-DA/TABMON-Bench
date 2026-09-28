"""Target-label-free source/target diagnostics for external validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def observable_domain_diagnostics(
    reference_X: pd.DataFrame,
    target_X: pd.DataFrame,
    model: Any,
    *,
    random_seed: int,
    max_rows: int = 2000,
    density_ratio_clip: float = 10.0,
) -> dict[str, float]:
    """Compute diagnostics using only reference/target features and outputs."""
    if reference_X.empty or target_X.empty:
        raise ValueError("Reference and target features must be non-empty")
    rng = np.random.default_rng(random_seed)
    n_reference = min(max_rows, len(reference_X))
    n_target = min(max_rows, len(target_X))
    reference = reference_X.iloc[
        rng.choice(len(reference_X), n_reference, replace=False)
    ].reset_index(drop=True)
    target = target_X.iloc[
        rng.choice(len(target_X), n_target, replace=False)
    ].reset_index(drop=True)

    combined = pd.concat([reference, target], ignore_index=True)
    encoded = pd.get_dummies(combined, dummy_na=True, dtype=float)
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    domain_y = np.concatenate(
        [np.zeros(n_reference, dtype=np.int8), np.ones(n_target, dtype=np.int8)]
    )
    train, test = train_test_split(
        np.arange(len(encoded)),
        test_size=0.3,
        random_state=random_seed,
        stratify=domain_y,
    )
    classifier = LogisticRegression(
        max_iter=250, solver="liblinear", random_state=random_seed
    ).fit(encoded.iloc[train], domain_y[train])
    domain_probability = classifier.predict_proba(encoded.iloc[test])[:, 1]
    domain_auc = float(roc_auc_score(domain_y[test], domain_probability))

    target_encoded = encoded.iloc[n_reference:]
    target_domain_probability = np.clip(
        classifier.predict_proba(target_encoded)[:, 1], 1e-6, 1.0 - 1e-6
    )
    ratio = target_domain_probability / (1.0 - target_domain_probability)
    clipped = np.minimum(ratio, density_ratio_clip)
    ess = (
        float(np.square(clipped.sum()) / np.square(clipped).sum())
        if np.square(clipped).sum() > 0
        else 0.0
    )

    reference_probability = np.asarray(model.predict_proba(reference), dtype=float)
    target_probability = np.asarray(model.predict_proba(target), dtype=float)

    def entropy(probability: np.ndarray) -> np.ndarray:
        safe = np.clip(probability, 1e-12, 1.0)
        return -np.sum(safe * np.log(safe), axis=1)

    reference_entropy = float(entropy(reference_probability).mean())
    target_entropy = float(entropy(target_probability).mean())
    reference_confidence = float(reference_probability.max(axis=1).mean())
    target_confidence = float(target_probability.max(axis=1).mean())

    violating_cells = 0
    violating_rows = np.zeros(len(target), dtype=bool)
    for column in reference.columns:
        source_values = reference[column]
        target_values = target[column]
        if pd.api.types.is_numeric_dtype(source_values.dtype):
            source_numeric = pd.to_numeric(source_values, errors="coerce").to_numpy(float)
            source_numeric = source_numeric[np.isfinite(source_numeric)]
            target_numeric = pd.to_numeric(target_values, errors="coerce").to_numpy(float)
            if len(source_numeric):
                violation = (
                    (target_numeric < source_numeric.min())
                    | (target_numeric > source_numeric.max())
                )
            else:
                violation = np.zeros(len(target), dtype=bool)
        else:
            known = set(source_values.astype("string").fillna("<NA>"))
            violation = ~target_values.astype("string").fillna("<NA>").isin(known).to_numpy()
        violating_cells += int(np.sum(violation))
        violating_rows |= np.asarray(violation, dtype=bool)

    return {
        "domain_classifier_auc": domain_auc,
        "effective_sample_size_fraction": ess / n_target,
        "density_ratio_clipping_rate": float(np.mean(ratio > density_ratio_clip)),
        "prediction_entropy_shift": target_entropy - reference_entropy,
        "prediction_confidence_shift": target_confidence - reference_confidence,
        "novel_support_rate": violating_cells / (len(target) * len(target.columns)),
        "novel_row_rate": float(violating_rows.mean()),
    }

