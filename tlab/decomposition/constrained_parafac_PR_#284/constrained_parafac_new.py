import numpy as np
import warnings

import tensorly as tl
from tensorly.random import random_cp
from tensorly.base import unfold
from tensorly.cp_tensor import (CPTensor, unfolding_dot_khatri_rao, cp_norm,
                                validate_cp_rank)
# Author: Jean Kossaifi
#         Jeremy Cohen <jeremy.cohen@irisa.fr>
#         Caglayan Tuna <caglayantun@gmail.com>

# License: BSD 3 clause


def initialize_constrained_parafac(tensor, rank, constraints, reg_par=None, prox_par=None, init='svd', svd='numpy_svd',
                                   random_state=None):
    r"""Initialize factors used in `constrained_parafac`.

    Parameters
    ----------

    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices with uniform distribution using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor. If init is a previously initialized `cp tensor`, all
    the weights are pulled in the last factor and then the weights are set to "1" for the output tensor.
    Lastly, factors are updated with proximal operator according to the selected constraint(s), so that they satisfy the
    imposed constraints (does not apply to cptensor initialization).

    Parameters
    ----------
    tensor : ndarray
    rank : int
    constraints : string
    reg_par : float
    prox_par : float
    random_state : {None, int, np.random.RandomState}
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS

    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    n_modes = tl.ndim(tensor)
    rng = tl.check_random_state(random_state)
    if constraints is None or len(constraints) != n_modes:
        constraints = [constraints] * n_modes
    if reg_par is None or len(reg_par) != n_modes:
        reg_par = [reg_par] * n_modes
    if prox_par is None or len(prox_par) != n_modes:
        prox_par = [prox_par] * n_modes

    if init == 'random':
        weights, factors = random_cp(tl.shape(tensor), rank, normalise_factors=False, **tl.context(tensor))

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, S, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank)

            # Put SVD initialization on the same scaling as the tensor in case normalize_factors=False
            if mode == 0:
                idx = min(rank, tl.shape(S)[0])
                U = tl.index_update(U, tl.index[:, :idx], U[:, :idx] * S[:idx])

            if tensor.shape[mode] < rank:
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])),
                                        **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)

            factors.append(U[:, :rank])

    elif isinstance(init, (tuple, list, CPTensor)):
        try:
            weights, factors = CPTensor(init)

            if tl.all(weights == 1):
                weights, factors = CPTensor((None, factors))
            else:
                weights_avg = tl.prod(weights) ** (1.0 / tl.shape(weights)[0])
                for i in range(len(factors)):
                    factors[i] = factors[i] * weights_avg
            kt = CPTensor((None, factors))
            return kt
        except ValueError:
            raise ValueError(
                'If initialization method is a mapping, then it must '
                'be possible to convert it to a CPTensor instance'
            )
    else:
        raise ValueError('Initialization method "{}" not recognized'.format(init))

    for i in range(n_modes):
        factors[i] = proximal_operator(factors[i], constraint=constraints[i], reg_par=reg_par[i], prox_par=prox_par[i])
    kt = CPTensor((None, factors))
    return kt


def initialize_constraints_and_parameters(n_modes, non_negative=None, l1_reg=None,
                                          l2_reg=None, l2_square=None, unimodality=None, normalize=None,
                                          simplex_par=None, normalized_sparsity_par=None,
                                          soft_sparsity_par=None, smoothness=None, monotonicity=None,
                                          hard_sparsity_par=None):
    """
    This function provides list of constraints and paremeters for constrained_parafac according to the user selection.

    Parameters
    ----------
    n_modes : int
    non_negative :
    l1_reg :
    l2_reg :
    l2_square :
    unimodality :
    normalize :
    simplex_par :
    normalized_sparsity_par :
    soft_sparsity_par :
    smoothness :
    monotonicity :
    hard_sparsity_par :

    Returns
    -------
    constraints : list of strings
    parameters: list of float
    """
    constraints = [None] * n_modes
    parameters = [None] * n_modes
    if non_negative:
        if isinstance(non_negative, dict):
            modes = list(non_negative)
            for i in range(len(modes)):
                constraints[modes[i]] = 'non_negative'
        else:
            for i in range(len(constraints)):
                constraints[i] = 'non_negative'
    if l1_reg:
        if isinstance(l1_reg, dict):
            modes = list(l1_reg)
            for i in range(len(modes)):
                constraints[modes[i]] = 'sparse_l1'
                parameters[modes[i]] = l1_reg[modes[i]]
        else:
            for i in range(len(constraints)):
                constraints[i] = 'sparse_l1'
                if isinstance(l1_reg, list):
                    parameters[i] = l1_reg[i]
                else:
                    parameters[i] = l1_reg
    if l2_reg:
        if isinstance(l2_reg, dict):
            modes = list(l2_reg)
            for i in range(len(modes)):
                constraints[modes[i]] = 'l2'
                parameters[modes[i]] = l2_reg[modes[i]]
        else:
            for i in range(len(constraints)):
                constraints[i] = 'l2'
                if isinstance(l2_reg, list):
                    parameters[i] = l2_reg[i]
                else:
                    parameters[i] = l2_reg

    if l2_square:
        if isinstance(l2_square, dict):
            modes = list(l2_square)
            for i in range(len(modes)):
                constraints[modes[i]] = 'l2_square'
                parameters[modes[i]] = l2_square[modes[i]]
        else:
            for i in range(len(constraints)):
                constraints[i] = 'l2_square'
                if isinstance(l2_square, list):
                    parameters[i] = l2_square[i]
                else:
                    parameters[i] = l2_square
    if normalized_sparsity_par:
        if isinstance(normalized_sparsity_par, dict):
            modes = list(normalized_sparsity_par)
            for i in range(len(modes)):
                constraints[modes[i]] = 'normalized_sparsity'
                parameters[modes[i]] = normalized_sparsity_par[modes[i]]
        else:
            for i in range(len(constraints)):
                constraints[i] = 'normalized_sparsity'
                if isinstance(normalized_sparsity_par, list):
                    parameters[i] = normalized_sparsity_par[i]
                else:
                    parameters[i] = normalized_sparsity_par
    if soft_sparsity_par:
        if isinstance(soft_sparsity_par, dict):
            modes = list(soft_sparsity_par)
            for i in range(len(modes)):
                constraints[modes[i]] = 'soft_sparsity'
                parameters[modes[i]] = soft_sparsity_par[modes[i]]
        else:
            for i in range(len(constraints)):
                constraints[i] = 'soft_sparsity'
                if isinstance(soft_sparsity_par, list):
                    parameters[i] = soft_sparsity_par[i]
                else:
                    parameters[i] = soft_sparsity_par
    if hard_sparsity_par:
        if isinstance(hard_sparsity_par, dict):
            modes = list(hard_sparsity_par)
            for i in range(len(modes)):
                constraints[modes[i]] = 'hard_sparsity'
                parameters[modes[i]] = hard_sparsity_par[modes[i]]
        else:
            for i in range(len(constraints)):
                constraints[i] = 'hard_sparsity'
                if isinstance(hard_sparsity_par, list):
                    parameters[i] = hard_sparsity_par[i]
                else:
                    parameters[i] = hard_sparsity_par
    if simplex_par:
        if isinstance(simplex_par, dict):
            modes = list(simplex_par)
            for i in range(len(modes)):
                constraints[modes[i]] = 'simplex'
                parameters[modes[i]] = simplex_par[modes[i]]
        else:
            for i in range(len(constraints)):
                constraints[i] = 'simplex'
                if isinstance(simplex_par, list):
                    parameters[i] = simplex_par[i]
                else:
                    parameters[i] = simplex_par
    if unimodality:
        if isinstance(unimodality, dict):
            modes = list(unimodality)
            for i in range(len(modes)):
                constraints[modes[i]] = 'unimodality'
        else:
            for i in range(len(constraints)):
                constraints[i] = 'unimodality'
    if monotonicity:
        if isinstance(monotonicity, dict):
            modes = list(monotonicity)
            for i in range(len(modes)):
                constraints[modes[i]] = 'monotonicity'
        else:
            for i in range(len(constraints)):
                constraints[i] = 'monotonicity'
    if normalize:
        if isinstance(normalize, dict):
            modes = list(normalize)
            for i in range(len(modes)):
                constraints[modes[i]] = 'normalize'
        else:
            for i in range(len(constraints)):
                constraints[i] = 'normalize'
    return constraints, parameters


def constrained_parafac_new(tensor, rank, n_iter_max=100, n_iter_max_inner=10,
                        init='svd', svd='numpy_svd',
                        tol_outer=1e-8, tol_inner=1e-6, random_state=None,
                        verbose=0, return_errors=False,
                        cvg_criterion='abs_rec_error',
                        fixed_modes=None, non_negative=False, l1_reg=False,
                        l2_reg=None, l2_square=None, unimodality=False, normalize=False,
                        simplex_par=None, normalized_sparsity_par=None, soft_sparsity_par=None,
                        smoothness=False, monotonicity=False, hard_sparsity_par=None):
    """CANDECOMP/PARAFAC decomposition via alternating optimization of
    alternating direction method of multipliers (AO-ADMM):

    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::
        tensor = [|weights; factors[0], ..., factors[-1] |],
    where factors are either penalized or constrained according to the user-defined constraint.

    In order to compute the factors efficiently, the ADMM algorithm
    introduces an auxilliary factor which is called factor_aux in the function.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration for outer loop
    n_iter_max_inner : int
        Number of iteration for inner loop
    init : {'svd', 'random', cptensor}, optional
        Type of factor matrix initialization. See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol_outer : float, optional
        (Default: 1e-8) Relative reconstruction error tolerance for outer loop. The
        algorithm is considered to have found a local minimum when the
        reconstruction error is less than `tol_outer`.
    tol_inner : float, optional
        (Default: 1e-6) Absolute reconstruction error tolerance for factor update during inner loop, i.e. ADMM optimization.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    non_negative : bool or dictionary
        This constraint is clipping negative values to '0'. If it is True non-negative constraint is applied to all modes.
    l1_reg : float or list or dictionary, optional
    l2_reg : float or list or dictionary, optional
    l2_square : float or list or dictionary, optional
    unimodality : bool or dictionary, optional
        If it is True unimodality constraint is applied to all modes.
    normalize : bool or dictionary, optional
        This constraint divides all the values by maximum value of the input array. If it is True normalize constraint
        is applied to all modes.
    simplex_par : float or list or dictionary, optional
    normalized_sparsity_par : float or list or dictionary, optional
    soft_sparsity_par : float or list or dictionary, optional
    smoothness : bool or dictionary, optional
    monotonicity : bool or dictionary, optional
    hard_sparsity_par : float or list or dictionary, optional
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion if `tol` is not None.
       If 'rec_error',  algorithm stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', algorithm terminates when `|previous rec_error - current rec_error| < tol`.
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications", SIAM
           REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    .. [2] Huang, Kejun, Nicholas D. Sidiropoulos, and Athanasios P. Liavas.
           "A flexible and efficient algorithmic framework for constrained matrix and tensor factorization." IEEE
           Transactions on Signal Processing 64.19 (2016): 5052-5065.
    """
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)
    n_modes = tl.ndim(tensor)
    constraints, parameters = initialize_constraints_and_parameters(n_modes, non_negative=non_negative, l1_reg=l1_reg,
                                                                    l2_reg=l2_reg, l2_square=l2_square,
                                                                    unimodality=unimodality, normalize=normalize,
                                                                    simplex_par=simplex_par,
                                                                    normalized_sparsity_par=normalized_sparsity_par,
                                                                    soft_sparsity_par=soft_sparsity_par,
                                                                    smoothness=smoothness, monotonicity=monotonicity,
                                                                    hard_sparsity_par=hard_sparsity_par)

    weights, factors = initialize_constrained_parafac(tensor, rank, constraints=constraints,
                                                      reg_par=reg_par,
                                                      prox_par=prox_par,
                                                      init=init, svd=svd,
                                                      random_state=random_state)

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    if fixed_modes is None:
        fixed_modes = []

    if tl.ndim(tensor) - 1 in fixed_modes:
        warnings.warn('You asked for fixing the last mode, which is not supported.\n '
                      'The last mode will not be fixed. Consider using tl.moveaxis()')
        fixed_modes.remove(tl.ndim(tensor) - 1)
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    # ADMM inits
    dual_var = []
    factors_aux = []
    for i in range(len(factors)):
        dual_var.append(tl.zeros(tl.shape(factors[i])))
        factors_aux.append(tl.transpose(tl.zeros(tl.shape(factors[i]))))

    for iteration in range(n_iter_max):
        if verbose > 1:
            print("Starting iteration", iteration + 1)
        for mode in modes_list:
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))

            pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse * tl.dot(tl.transpose(factor), factor)

            mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

            factors[mode], factors_aux[mode], dual_var[mode] = admm(mttkrp, pseudo_inverse, factors[mode], dual_var[mode],
                                                                    n_iter_max=n_iter_max_inner, constraint=constraints[mode],
                                                                    reg_par=reg_par[mode], prox_par=prox_par[mode], tol=tol_inner)

        factors_norm = cp_norm((weights, factors))
        iprod = tl.sum(tl.sum(mttkrp * factors[-1], axis=0) * weights)
        rec_error = tl.sqrt(tl.abs(norm_tensor ** 2 + factors_norm ** 2 - 2 * iprod)) / norm_tensor
        rec_errors.append(rec_error)
        constraint_error = 0
        for mode in modes_list:
            constraint_error += tl.norm(factors[mode] - tl.transpose(factors_aux[mode])) / tl.norm(factors[mode])
        if tol_outer:

            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print("iteration {}, reconstruction error: {}, decrease = {}, unnormalized = {}".format(iteration,
                                                                                                            rec_error,
                                                                                                            rec_error_decrease))

                if constraint_error < tol_outer:
                    break
                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol_outer
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol_outer
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print("PARAFAC converged after {} iterations".format(iteration))
                    break

            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    cp_tensor = CPTensor((weights, factors))

    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor