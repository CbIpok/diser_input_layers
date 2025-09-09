import os
import json
import unittest
import numpy as np

from plot_basis_maps import load_basis_coofs, load_basis, calc_mse


class TestIntegrationCalcMSE(unittest.TestCase):
    def test_calc_mse_real_data_slice(self):
        # Use i=16 because both data/basis_16 and coefs_process/basis_16.json exist
        i = 16
        coefs_json = os.path.join('coefs_process', f'basis_{i}.json')
        basis_dir = os.path.join('data', f'basis_{i}')

        # Sanity check presence
        self.assertTrue(os.path.exists(coefs_json))
        self.assertTrue(os.path.isdir(basis_dir))
        self.assertTrue(os.path.exists('data/functions.wave'))
        self.assertTrue(os.path.exists('data/config.json'))

        xs, ys, approx_err, coefs = load_basis_coofs(coefs_json)

        # Reduce size for quicker test: take first 250 samples
        m = min(250, xs.size)
        xs = xs[:m]
        ys = ys[:m]
        approx_err = approx_err[:m]
        coefs = [c[:m] for c in coefs]

        basis = np.array(load_basis(basis_dir))
        to_restore = np.loadtxt('data/functions.wave')

        mse = calc_mse(to_restore, basis, xs, ys, coefs)
        # Shape must match scene
        with open('data/config.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.assertEqual(mse.shape, (cfg['size']['y'], cfg['size']['x']))
        # There should be finite values at the provided coordinates (or their subset)
        nn = np.isfinite(mse[tuple(xs.astype(int)), tuple(ys.astype(int))]).sum()
        self.assertGreater(nn, 0)


if __name__ == '__main__':
    unittest.main()

