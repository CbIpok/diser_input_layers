import unittest
import numpy as np

from diser.core.restore import (
    reconstruct_from_bases,
    valid_mask_from_bases,
    mse_on_valid_region,
    pointwise_rmse_from_coefs,
)


class TestDiserCoreRestore(unittest.TestCase):
    def test_reconstruct_and_mse(self):
        H, W, k = 5, 4, 3
        rng = np.random.default_rng(0)
        bases = rng.normal(size=(k, H, W))
        bases[:, 0, 0] = np.nan  # introduce NaN
        c = np.array([0.5, -2.0, 1.0])
        Z_hat = reconstruct_from_bases(c, bases)
        self.assertEqual(Z_hat.shape, (H, W))
        vm = valid_mask_from_bases(bases)
        self.assertTrue(vm.any())
        mse = mse_on_valid_region(Z_hat, Z_hat, vm)
        self.assertAlmostEqual(mse, 0.0, places=12)

    def test_pointwise_rmse(self):
        H, W, k = 4, 4, 2
        b0 = np.zeros((H, W)); b0[1, 1] = 1.0
        b1 = np.zeros((H, W)); b1[2, 2] = 1.0
        bases = np.stack([b0, b1], axis=0)
        xs = np.array([1]); ys = np.array([1])
        coefs = [np.array([2.0]), np.array([0.0])]
        target = 2.0 * b0
        rmse = pointwise_rmse_from_coefs(target, bases, xs, ys, coefs)
        self.assertEqual(rmse.shape, (H, W))
        self.assertAlmostEqual(rmse[1, 1], 0.0)


if __name__ == '__main__':
    unittest.main()
