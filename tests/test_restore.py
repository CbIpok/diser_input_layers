import unittest
import numpy as np

from utils.restore import (
    approximate_with_non_orthogonal_basis,
    approximate_with_non_orthogonal_basis_orto,
)


class TestRestore(unittest.TestCase):
    def test_least_squares_reconstruction(self):
        # Basis: 3 vectors in R^5
        rng = np.random.default_rng(0)
        basis = [rng.normal(size=5) for _ in range(3)]
        true_c = np.array([0.5, -1.2, 2.0])
        A = np.column_stack(basis)
        v = A @ true_c

        approx, coefs = approximate_with_non_orthogonal_basis(v, basis)
        np.testing.assert_allclose(approx, v, atol=1e-10)
        np.testing.assert_allclose(coefs, true_c, atol=1e-10)

    def test_ortho_method_matches_projection(self):
        rng = np.random.default_rng(1)
        basis = [rng.normal(size=8) for _ in range(4)]
        true_c = np.array([1.0, 0.0, -0.5, 2.0])
        v = np.column_stack(basis) @ true_c

        approx, coefs = approximate_with_non_orthogonal_basis_orto(v, basis)
        self.assertIsNotNone(approx)
        self.assertIsNotNone(coefs)
        np.testing.assert_allclose(approx, v, atol=1e-10)
        np.testing.assert_allclose(coefs, true_c, atol=1e-8)


if __name__ == '__main__':
    unittest.main()

