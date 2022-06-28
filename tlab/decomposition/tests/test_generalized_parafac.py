from .._generalized_parafac import (generalized_parafac, loss_operator_func, gradient_operator_func,
                                    vectorize_factors, vectorized_factors_to_tensor, vectorized_mttkrp, GCP)
from tensorly.testing import assert_, assert_array_equal, assert_class_wrapper_correctly_passes_arguments
from tensorly.cp_tensor import cp_to_tensor
from tensorly.random import random_cp
import tensorly as tl


def test_loss_operator_func():
    """Test for loss operator"""

    tensor = tl.tensor([1, 0, 2, 2])

    function = loss_operator_func(tensor, rank=3, loss="gaussian")
    assert_(callable(function) == True)


def test_gradient_operator_func():
    """Test for gradient operator"""
    tensor = tl.tensor([1, 0, 2, 2])

    # Gaussian gradient
    function = gradient_operator_func(tensor, rank=3, loss="gaussian")
    assert_(callable(function) == True)


def test_vectorize_factors():
    """Test for the vectorized_factors
    """
    shape = [8, 10, 6]
    rank = 3
    weights, factors = random_cp(shape, rank)
    vec_factors = vectorize_factors(factors)
    assert_(len(vec_factors) == tl.sum(tl.tensor(shape)) * rank)


def test_vectorized_factors_to_tensor():
    """Test for the vectorized_factors_to_tensor
    """
    shape = [8, 10, 6]
    rank = 3
    weights, factors = random_cp(shape, rank, normalise_factors=False)
    tensor = tl.cp_to_tensor((weights, factors))
    vec_factors = vectorize_factors(factors)
    rec_tensor = vectorized_factors_to_tensor(vec_factors, shape, rank)
    assert_array_equal(tensor, rec_tensor)


def test_vectorized_mttkrp():
    """Test for the vectorized_mttkrp
    """
    shape = [8, 10, 6]
    rank = 3
    weights, factors = random_cp(shape, rank)
    vec_factors = vectorize_factors(factors)
    tensor = tl.cp_to_tensor((weights, factors))
    vec_mttkrp = vectorized_mttkrp(tensor, vec_factors, rank)
    assert_(tl.shape(vec_mttkrp) == tl.shape(vec_factors))


def test_generalized_parafac(monkeypatch):
    """Test for the Generalized Parafac decomposition
    """
    tol_norm_2 = 0.3
    rank = 3
    shape = [8, 10, 6]
    init = 'random'
    rng = tl.check_random_state(1234)
    initial_tensor = cp_to_tensor(random_cp(shape, rank=rank))

    # Gaussian
    loss = 'gaussian'
    array = rng.normal(initial_tensor)
    tensor = tl.tensor(array)
    _, factors = generalized_parafac(tensor, loss=loss, rank=rank, init=init)
    vectorized_factors = vectorize_factors(factors)
    fun_error = loss_operator_func(initial_tensor, rank, loss)
    error = fun_error(vectorized_factors) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Gamma
    loss = 'gamma'
    array = rng.gamma(1, initial_tensor, size=shape)
    tensor = tl.tensor(array)
    _, factors = generalized_parafac(tensor, loss=loss, rank=rank, init=init)
    vectorized_factors = vectorize_factors(factors)
    fun_error = loss_operator_func(initial_tensor, rank, loss)
    error = fun_error(vectorized_factors) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Rayleigh
    loss = 'rayleigh'
    array = rng.rayleigh(initial_tensor, size=shape)
    tensor = tl.tensor(array)
    _, factors = generalized_parafac(tensor, loss=loss, rank=rank, init=init)
    vectorized_factors = vectorize_factors(factors)
    fun_error = loss_operator_func(initial_tensor, rank, loss)
    error = fun_error(vectorized_factors) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Poisson-count
    loss = 'poisson_count'
    array = 1.0 * rng.poisson(initial_tensor, size=shape)
    tensor = tl.tensor(array)
    _, factors = generalized_parafac(tensor, loss=loss, rank=rank, init=init)
    vectorized_factors = vectorize_factors(factors)
    fun_error = loss_operator_func(initial_tensor, rank, loss)
    error = fun_error(vectorized_factors) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Poisson-log
    loss = 'poisson_log'
    array = 1.0 * rng.poisson(tl.exp(initial_tensor), size=shape)
    tensor = tl.tensor(array)
    _, factors = generalized_parafac(tensor, loss=loss, rank=rank, init=init)
    vectorized_factors = vectorize_factors(factors)
    fun_error = loss_operator_func(initial_tensor, rank, loss)
    error = fun_error(vectorized_factors) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Bernoulli-odds
    loss = 'bernoulli_odds'
    shape = [30, 40, 20]
    rank = 10
    initial_tensor = cp_to_tensor(random_cp(shape, rank=rank))
    array = 1.0 * rng.binomial(1, initial_tensor / (initial_tensor + 1), size=shape)
    tensor = tl.tensor(array)
    _, factors = generalized_parafac(tensor, loss=loss, rank=rank, init=init)
    vectorized_factors = vectorize_factors(factors)
    fun_error = loss_operator_func(initial_tensor, rank, loss)
    error = fun_error(vectorized_factors) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Bernoulli-logit
    loss = 'bernoulli_logit'
    array = 1.0 * rng.binomial(1, tl.exp(initial_tensor) / (tl.exp(initial_tensor) + 1), size=shape)
    tensor = tl.tensor(array)
    _, factors = generalized_parafac(tensor, loss=loss, rank=rank, init=init)
    vectorized_factors = vectorize_factors(factors)
    fun_error = loss_operator_func(initial_tensor, rank, loss)
    error = fun_error(vectorized_factors) / tl.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')
    assert_class_wrapper_correctly_passes_arguments(monkeypatch, generalized_parafac, GCP, rank=3)
