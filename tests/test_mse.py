import json
import os
import tempfile
import unittest
import numpy as np

from plot_basis_maps import calc_mse


class TestMSEFormula(unittest.TestCase):
    def test_mse_zero_on_exact_composition(self):
        # Build tiny grid 4x4 with 2 basis functions
        H, W, k = 4, 4, 2
        b0 = np.zeros((H, W))
        b1 = np.zeros((H, W))
        # Activate a few positions
        b0[1, 1] = 1.0
        b0[2, 2] = 0.5
        b1[1, 1] = 2.0
        b1[2, 2] = -1.0
        basis = np.stack([b0, b1], axis=0)

        # Compose target as c0*b0 + c1*b1 at two points
        xs = np.array([1, 2])
        ys = np.array([1, 2])
        coefs = [np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        to_restore = coefs[0][0] * b0 + coefs[1][0] * b1  # dummy full grid

        # Create temp data/config.json and MOST bath file matching H,W
        with tempfile.TemporaryDirectory() as td:
            data_dir = os.path.join(td, 'data')
            bath_dir = os.path.join(data_dir, 'bath')
            os.makedirs(bath_dir, exist_ok=True)
            bath_path = os.path.join(bath_dir, 'tiny')
            # MOST format: nx ny, then nx x-coords, ny y-coords, then values (we don't care values)
            nx, ny = W, H
            xs_line = ' '.join(str(i) for i in range(nx))
            ys_line = ' '.join(str(j) for j in range(ny))
            vals = ' '.join('0' for _ in range(nx * ny))
            with open(bath_path, 'w', encoding='utf-8') as f:
                f.write(f"{nx} {ny}\n{xs_line}\n{ys_line}\n{vals}\n")
            cfg = {"size": {"x": nx, "y": ny}, "bath_path": bath_path}
            with open(os.path.join(data_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False)

            # Run inside temp dir so calc_mse finds data/config.json
            cwd = os.getcwd()
            try:
                os.chdir(td)
                mse = calc_mse(to_restore, basis, xs, ys, coefs)
            finally:
                os.chdir(cwd)

        self.assertEqual(mse.shape, (H, W))
        self.assertAlmostEqual(mse[1, 1], 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
