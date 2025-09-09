import json
import os
import tempfile
import unittest
import numpy as np

from plot_basis_maps import load_basis_coofs as load_basis_coefs_plot
from point import load_basis_coefs as load_basis_coefs_point


SAMPLE = {
    "(10, 5)": {"coefs": [1.0, 2.0, 3.0], "aprox_error": 0.1},
    "(11, 6)": {"coefs": [4.0, 5.0, 6.0], "aprox_error": 0.2},
}


class TestCoeffsIO(unittest.TestCase):
    def _write_json(self, td):
        path = os.path.join(td, 'basis_3.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(SAMPLE, f, ensure_ascii=False)
        return path

    def test_plot_parser(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_json(td)
            xs, ys, approx, coefs = load_basis_coefs_plot(path)
            self.assertEqual(xs.shape, (2,))
            self.assertEqual(len(coefs), 3)
            np.testing.assert_allclose(approx, [0.1, 0.2])

    def test_point_parser(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_json(td)
            xs, ys, coefs = load_basis_coefs_point(path)
            self.assertEqual(xs.shape, (2,))
            self.assertEqual(len(coefs), 3)
            self.assertEqual(coefs[0].shape, (2,))


if __name__ == '__main__':
    unittest.main()

