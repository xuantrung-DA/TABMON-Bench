import inspect
import unittest

import numpy as np

from src.evaluation.natural_shift_protocol import (
    MONITOR_OUTPUT_COLUMNS,
    assert_disjoint_complete,
    select_largest_groups,
    split_source_indices,
    temporal_source_and_windows,
    validate_monitor_output_columns,
)
from src.evaluation.observable_diagnostics import observable_domain_diagnostics


class NaturalShiftProtocolTest(unittest.TestCase):
    def test_temporal_target_is_late_contiguous_and_complete(self):
        source, windows = temporal_source_and_windows(100, 0.30, 5)
        self.assertTrue(np.array_equal(source, np.arange(70)))
        self.assertTrue(np.array_equal(np.concatenate(windows), np.arange(70, 100)))
        self.assertTrue(all(np.all(np.diff(window) == 1) for window in windows))
        assert_disjoint_complete(np.arange(100), [source, *windows])

    def test_largest_group_selection_has_no_outcome_argument(self):
        groups = ["a"] * 4 + ["b"] * 6 + ["c"] * 5
        self.assertEqual(select_largest_groups(groups, 2), ["b", "c"])
        parameters = inspect.signature(select_largest_groups).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("y", parameters)

    def test_source_split_is_disjoint_complete_and_stratified(self):
        positions = np.arange(100)
        labels = np.asarray([0, 1] * 50)
        train, calibration, holdout = split_source_indices(positions, labels, 42)
        assert_disjoint_complete(positions, [train, calibration, holdout])
        self.assertEqual((len(train), len(calibration), len(holdout)), (60, 20, 20))
        for part in (train, calibration, holdout):
            self.assertAlmostEqual(float(labels[part].mean()), 0.5)

    def test_monitor_output_schema_is_an_exact_allowlist(self):
        validate_monitor_output_columns(MONITOR_OUTPUT_COLUMNS)
        with self.assertRaises(ValueError):
            validate_monitor_output_columns((*MONITOR_OUTPUT_COLUMNS, "target_label"))

    def test_diagnostics_api_has_no_target_label_parameter(self):
        parameters = {
            name.lower()
            for name in inspect.signature(observable_domain_diagnostics).parameters
        }
        self.assertTrue(
            parameters.isdisjoint({"label", "labels", "target_y", "y_true", "oracle"})
        )


if __name__ == "__main__":
    unittest.main()

