import os
import unittest
import numpy as np

from diser.io.coeffs import resolve_coeffs_dir
from point import (
    load_basis_coefs,
    load_basis_dir,
    get_coeffs_for_point,
    reconstruct_from_bases,
    valid_mask_from_bases,
    mse_on_valid_region,
)


class TestIntegrationPointPipeline(unittest.TestCase):
    def test_point_reconstruction_mse_nonnegative(self):
        i = 16
        coefs_dir = resolve_coeffs_dir('coefs_process', 'data/functions.wave')
        coefs_json = coefs_dir / f'basis_{i}.json'
        basis_dir = os.path.join('data', f'basis_{i}')

        self.assertTrue(coefs_json.exists())
        self.assertTrue(os.path.isdir(basis_dir))
        self.assertTrue(os.path.exists('data/functions.wave'))

        xs, ys, coefs = load_basis_coefs(coefs_json)
        bases = load_basis_dir(basis_dir)
        to_restore = np.loadtxt('data/functions.wave')

        # Take a reasonable point — nearest to default (100, 547)
        c, idx = get_coeffs_for_point(100, 547, xs, ys, coefs)
        Z_hat = reconstruct_from_bases(c, bases)
        valid_mask = valid_mask_from_bases(bases)
        mse_val = mse_on_valid_region(to_restore, Z_hat, valid_mask)

        self.assertTrue(np.isfinite(mse_val))
        self.assertGreaterEqual(mse_val, 0.0)


if __name__ == '__main__':
    unittest.main()

