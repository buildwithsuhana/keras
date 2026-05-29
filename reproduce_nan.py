import numpy as np
import unittest

class TestNaN(unittest.TestCase):
    def test_nan(self):
        final_loss = np.nan
        self.assertTrue(np.isfinite(final_loss))

if __name__ == "__main__":
    unittest.main()
