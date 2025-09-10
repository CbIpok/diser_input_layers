import numpy as np

# Thin wrappers delegating to the new modular implementations
from diser.core import restore as _core_restore

def approximate_with_non_orthogonal_basis(vector, basis):
    # Least squares using column-stacked basis
    A = np.column_stack(basis)
    coefs, *_ = np.linalg.lstsq(A, vector, rcond=None)
    approx = A @ coefs
    return approx, coefs

def approximate_with_non_orthogonal_basis_orto(vector, basis):
    # Reuse least-squares path to avoid duplicating Gram-Schmidt code
    return approximate_with_non_orthogonal_basis(vector, basis)
