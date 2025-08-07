import copy
import numpy as np



### PUBLIC ###

def approximate_with_non_orthogonal_basis(vector, basis):
    vector = copy.deepcopy(vector)
    basis = copy.deepcopy(basis)
    A = np.column_stack(basis)
    coefficients, residuals, _, _ = np.linalg.lstsq(A, vector, rcond=None)
    approximation = np.dot(A, coefficients)
    return approximation, coefficients



def approximate_with_non_orthogonal_basis_orto(vector, basis):
    # Убираем глубокое копирование
    bort = _gram_schmidt_with_fixed_first_vector(basis)
    f_bort = _decompose_vector(vector, bort)

    try:
        # Оптимизируем нахождение коэффициентов
        coofs = _find_coefficients_in_original_basis(basis, bort, f_bort)
    except np.linalg.LinAlgError:
        return None, None

    # Оптимизированное вычисление аппроксимации
    aprox = np.dot(coofs, np.array(basis))

    return aprox, coofs


### PRIVATE ###

def _gram_schmidt_with_fixed_first_vector(vectors):
    """Ортогонализация системы векторов методом Грама-Шмидта, фиксируя первый вектор."""
    orthogonal_basis = [vectors[0].astype(float)]  # Первый вектор остаётся неизменным
    for v in vectors[1:]:
        v = v.astype(float)
        for u in orthogonal_basis:
            # Оптимизированная операция вычитания с использованием NumPy
            v -= np.dot(v, u) / np.dot(u, u) * u
        orthogonal_basis.append(v)
    return orthogonal_basis


def _decompose_vector(vector, basis):
    """Находит коэффициенты разложения вектора по базису."""
    # Используем векторизацию с NumPy для вычисления коэффициентов
    basis_norms = np.array([np.dot(b, b) for b in basis])
    coefficients = np.dot(vector, np.array(basis).T) / basis_norms
    return coefficients


def _find_coefficients_in_original_basis(basis, orthogonal_basis, f_bort):
    """Находит коэффициенты разложения вектора по исходному базису из ортогонализованного."""
    # Создание матрицы перехода с использованием NumPy
    transition_matrix = np.dot(basis, np.array(orthogonal_basis).T)

    # Решение системы уравнений
    norm_factors = np.array([np.dot(ob, ob) for ob in orthogonal_basis])
    f_b = np.linalg.solve(transition_matrix.T, f_bort * norm_factors)

    return f_b

