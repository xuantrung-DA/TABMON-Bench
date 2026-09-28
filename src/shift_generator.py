import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple


class TabularShiftGenerator:
    def __init__(
        self,
        test_pool_df: pd.DataFrame,
        target_col: str,
        random_seed: int = 42,
    ):
        self.df = test_pool_df.copy()
        self.target_col = target_col
        # Independent random streams keep pre-shift batches identical across
        # shift families/severities for the same experimental seed.
        shift_seed, stream_seed = np.random.SeedSequence(random_seed).spawn(2)
        self.shift_rng = np.random.default_rng(shift_seed)
        self.stream_rng = np.random.default_rng(stream_seed)
        self.features = [c for c in self.df.columns if c != target_col]

        self.severity_map = {"low": 0.5, "medium": 1.5, "high": 3.0}

    def _rejection_sampling(
        self, data: pd.DataFrame, weights: np.ndarray
    ) -> pd.DataFrame:
        """Resample rows with replacement according to normalized weights."""
        if data.empty:
            raise ValueError("Cannot sample from an empty dataframe")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(data),) or not np.all(np.isfinite(weights)):
            raise ValueError("Shift weights must be finite and match the data length")
        total = weights.sum()
        if total <= 0:
            raise ValueError("Shift weights must have a positive sum")
        sampled_positions = self.shift_rng.choice(
            len(data), size=len(data), replace=True, p=weights / total
        )
        return data.iloc[sampled_positions].copy()

    def _create_stream(
        self,
        base_df: pd.DataFrame,
        shifted_df: pd.DataFrame,
        mode: str,
        num_batches: int,
        batch_size: int,
    ) -> List[pd.DataFrame]:
        """Create an abrupt, gradual, or fully shifted batch stream."""
        if num_batches < 1 or batch_size < 1:
            raise ValueError("num_batches and batch_size must both be positive")
        if mode not in {"abrupt", "gradual", "static"}:
            raise ValueError(f"Unknown stream mode: {mode}")

        stream = []
        for i in range(num_batches):
            if mode == "abrupt":
                alpha = 0.0 if i < num_batches // 2 else 1.0
            elif mode == "gradual":
                alpha = i / (num_batches - 1) if num_batches > 1 else 1.0
            else:
                alpha = 1.0

            n_shifted = int(batch_size * alpha)
            n_base = batch_size - n_shifted

            batch_base = (
                base_df.sample(
                    n=n_base,
                    replace=True,
                    random_state=int(self.stream_rng.integers(1e5)),
                )
                if n_base > 0
                else pd.DataFrame()
            )
            batch_shift = (
                shifted_df.sample(
                    n=n_shifted,
                    replace=True,
                    random_state=int(self.stream_rng.integers(1e5)),
                )
                if n_shifted > 0
                else pd.DataFrame()
            )

            batch = pd.concat([batch_base, batch_shift]).sample(
                frac=1.0,
                random_state=int(self.stream_rng.integers(1e5)),
            )
            stream.append(batch)

        return stream

    def shift_b0_no_shift(
        self,
        num_batches: int = 10,
        batch_size: int = 1000,
    ) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B0: Null control sampled from the unmodified test distribution."""
        stream = self._create_stream(
            self.df,
            self.df,
            mode="static",
            num_batches=num_batches,
            batch_size=batch_size,
        )
        return stream, {feature: 0.0 for feature in self.features}

    def shift_b1_single_covariate(
        self,
        feature: str,
        severity: str,
        mode: str = "abrupt",
        num_batches: int = 10,
        batch_size: int = 1000,
    ) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B1: Resample by an exponential tilt of one numeric feature."""
        s_val = self.severity_map[severity]
        if feature not in self.features:
            raise ValueError(f"Unknown shift feature: {feature}")
        if not pd.api.types.is_numeric_dtype(self.df[feature]):
            raise TypeError(f"Covariate tilt requires a numeric feature, got: {feature}")
        x_val = self.df[feature].values

        norm_x = (x_val - np.mean(x_val)) / (np.std(x_val) + 1e-9)
        weights = np.exp(np.clip(s_val * norm_x, -50.0, 50.0))

        shifted_df = self._rejection_sampling(self.df, weights)
        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)

        gt = {f: 0.0 for f in self.features}
        gt[feature] = 1.0
        return stream, gt

    def shift_b2_correlated_multi(
        self,
        features: List[str],
        severity: str,
        mode: str = "abrupt",
        num_batches: int = 10,
        batch_size: int = 1000,
    ) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B2: Resample by an exponential tilt of multiple numeric features."""
        s_val = self.severity_map[severity]
        if not features:
            raise ValueError("At least one feature is required")

        combined_x = np.zeros(len(self.df))
        for f in features:
            if f not in self.features or not pd.api.types.is_numeric_dtype(self.df[f]):
                raise TypeError(f"Correlated tilt requires numeric feature: {f}")
            x_val = self.df[f].values
            combined_x += (x_val - np.mean(x_val)) / (np.std(x_val) + 1e-9)

        weights = np.exp(np.clip(s_val * combined_x / len(features), -50.0, 50.0))
        shifted_df = self._rejection_sampling(self.df, weights)
        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)

        gt = {f: 0.0 for f in self.features}
        for f in features:
            gt[f] = 1.0 / len(features)
        return stream, gt

    def shift_b4_support_violation(
        self,
        feature: str,
        severity: str,
        mode: str = "abrupt",
        num_batches: int = 10,
        batch_size: int = 1000,
    ) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B4: Translate one numeric feature beyond its observed support."""
        shifted_df = self.df.copy()
        if feature not in self.features or not pd.api.types.is_numeric_dtype(
            shifted_df[feature]
        ):
            raise TypeError(f"Support violation requires numeric feature: {feature}")

        shift_multiplier = (
            1.2 if severity == "low" else 1.5 if severity == "medium" else 2.0
        )
        max_val = shifted_df[feature].max()
        std_val = shifted_df[feature].std()

        shifted_df[feature] = shifted_df[feature] + shift_multiplier * (max_val + std_val)

        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)

        gt = {f: 0.0 for f in self.features}
        gt[feature] = 1.0
        return stream, gt

    def shift_b5_pipeline_corruption(
        self,
        feature: str,
        severity: str,
        mode: str = "abrupt",
        num_batches: int = 10,
        batch_size: int = 1000,
    ) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B5: Replace a severity-dependent fraction with sentinel -999."""
        shifted_df = self.df.copy()
        if feature not in self.features or not pd.api.types.is_numeric_dtype(
            shifted_df[feature]
        ):
            raise TypeError(f"Pipeline corruption requires numeric feature: {feature}")

        corrupt_rate = (
            0.2 if severity == "low" else 0.5 if severity == "medium" else 0.8
        )

        corrupt_mask = self.shift_rng.random(len(shifted_df)) < corrupt_rate
        shifted_df.loc[corrupt_mask, feature] = -999

        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)

        gt = {f: 0.0 for f in self.features}
        gt[feature] = 1.0
        return stream, gt

    def shift_b6_concept_shift_negative_control(
        self,
        condition_func: Callable,
        severity: str,
        mode: str = "abrupt",
        num_batches: int = 10,
        batch_size: int = 1000,
    ) -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """B6: Relabel a fixed feature region without modifying its inputs."""
        shifted_df = self.df.copy()

        flip_prob = (
            0.2 if severity == "low" else 0.5 if severity == "medium" else 1.0
        )

        mask = condition_func(shifted_df)
        flip_mask = mask & (self.shift_rng.random(len(shifted_df)) < flip_prob)

        labels = np.sort(pd.unique(shifted_df[self.target_col]))
        if len(labels) < 2:
            raise ValueError("Concept shift requires at least two target classes")
        if len(labels) == 2 and set(labels.tolist()) == {0, 1}:
            # Preserve the frozen binary generator exactly.
            shifted_df.loc[flip_mask, self.target_col] = (
                1 - shifted_df.loc[flip_mask, self.target_col]
            )
        else:
            # Deterministic cyclic relabeling gives every selected multiclass
            # observation a different label without changing X.
            successor = {
                label: labels[(position + 1) % len(labels)]
                for position, label in enumerate(labels)
            }
            shifted_df.loc[flip_mask, self.target_col] = shifted_df.loc[
                flip_mask, self.target_col
            ].map(successor)

        stream = self._create_stream(self.df, shifted_df, mode, num_batches, batch_size)

        # Observable feature attribution is undefined for label-only changes.
        gt = {f: 0.0 for f in self.features}
        return stream, gt
