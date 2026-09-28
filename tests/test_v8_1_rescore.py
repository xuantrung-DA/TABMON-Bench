import unittest

import numpy as np

from src.evaluation.risk import BINARY_LOG_LOSS_EPSILON, binary_log_losses


class V81RiskDefinitionTest(unittest.TestCase):
    def test_extreme_probabilities_use_frozen_v7_clipping(self):
        probabilities = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        losses = binary_log_losses(probabilities, [0, 1], [1, 0])
        expected = -np.log(BINARY_LOG_LOSS_EPSILON)
        np.testing.assert_allclose(losses, [expected, expected])

    def test_unknown_class_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unknown target class"):
            binary_log_losses(np.asarray([[0.5, 0.5]]), [0, 1], [2])


if __name__ == "__main__":
    unittest.main()
