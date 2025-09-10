import os
import tempfile
import unittest
import numpy as np

from diser.io.basis import save_basis_masks, load_basis_dir


class TestDiserBasisIO(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            H, W = 6, 7
            masks = []
            m0 = np.zeros((H, W), dtype=bool)
            m0[2:4, 3:5] = True
            m1 = np.zeros((H, W), dtype=float)
            m1[1, 1] = 1.0
            m1[5, 6] = 2.0
            masks.extend([m0, m1])

            save_basis_masks(masks, td)
            B = load_basis_dir(td)
            self.assertEqual(B.shape, (2, H, W))
            # Values restored as NaN outside mask, 1.0 inside
            self.assertTrue(np.isfinite(B[0][2, 3]))
            self.assertTrue(np.isnan(B[0][0, 0]))


if __name__ == '__main__':
    unittest.main()

