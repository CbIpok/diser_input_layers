import os
import tempfile
import unittest
import numpy as np

from point import load_basis_dir


class TestBasisIO(unittest.TestCase):
    def test_load_basis_dir_reads_and_sorts(self):
        with tempfile.TemporaryDirectory() as td:
            H, W = 5, 4
            # Create unsorted files: 2, 0, 1
            arrays = [
                (2, np.full((H, W), 3.0)),
                (0, np.full((H, W), 1.0)),
                (1, np.full((H, W), 2.0)),
            ]
            for idx, arr in arrays:
                np.savetxt(os.path.join(td, f"basis_{idx}.wave"), arr, fmt='%.1f')

            B = load_basis_dir(td)
            self.assertEqual(B.shape, (3, H, W))
            self.assertTrue(np.isnan(B[0, 0, 0]) or B[0, 0, 0] == 1.0)  # allow NaN mapping
            # Check ordering by verifying the distinct constant planes
            self.assertAlmostEqual(np.nanmean(B[0]), 1.0)
            self.assertAlmostEqual(np.nanmean(B[1]), 2.0)
            self.assertAlmostEqual(np.nanmean(B[2]), 3.0)


if __name__ == '__main__':
    unittest.main()

