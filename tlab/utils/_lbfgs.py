import tensorly as tl
import scipy
import numpy as np


def lbfgs(loss, x0, gradient=None, n_iter_max=100, non_negative=False, norm=1.0):
    """
    LBFGS optimizer to solve GCP decomposition.

    Parameters
    ----------
    loss : callable
    x0 : 1d ndarray
    gradient : callable
        Default : None
    n_iter_max : int
        Default : 100
    non_negative : bool
        Default : False
    norm : float
        Default : 1.0

    Returns
    ----------
    ndarray
    list of errors per iteration

    Notes
    --------
    Content of this function could be useful for Tensorly developers to adapt GCP to tensorly for all backends.
    Currently, it supports only numpy backend, but it could be improved for other backends as well.

    * pytorch: imposing non-negative constraint not good also it fails for some losses because of the backward part, probably.
    * tensorflow: There is no option for constrained loss, and it doesn't return error per iteration. Besides, tensorflow_probability library is a dependency.
    * jax: jax.scipy.minimize only supports bfgs according to website (https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.optimize.minimize.html). However, there are some attempts to add lbfgs (https://github.com/google/jax/pull/6053).
    * mxnet: This issue can be followed. https://github.com/apache/incubator-mxnet/issues/9182
    """

    if tl.get_backend() == "numpy":
        from scipy.optimize import minimize
        if non_negative:
            bound = scipy.optimize.Bounds(0, np.inf)
        else:
            bound = scipy.optimize.Bounds(-np.inf, np.inf)
        error = []
        error_func = lambda x: error.append(loss(x) / norm)
        return minimize(loss, x0, method='L-BFGS-B', jac=gradient, callback=error_func, options={'maxiter': n_iter_max}, bounds=bound).x, error

    elif tl.get_backend() == "pytorch":
        import torch
        x0.requires_grad = True
        optimizer = torch.optim.LBFGS([x0], history_size=10, max_iter=4, line_search_fn="strong_wolfe")
        error = []
        for i in range(n_iter_max):
            def closure():
                # Zero gradients
                optimizer.zero_grad()

                # Compute loss
                objective = loss(x0)

                # Backward pass
                objective.backward()

                return objective

            if non_negative:
                with torch.no_grad():
                    x0.clamp(min=0)
            optimizer.step(closure)
            objective = closure()
            error.append(objective.item() / norm)
        return x0, error

    elif tl.get_backend() == "tensorflow":
        import tensorflow_probability as tfp

        def quadratic_loss_and_gradient(x):
            return tfp.math.value_and_gradient(loss, x)
        error = []
        optim_results = tfp.optimizer.lbfgs_minimize(quadratic_loss_and_gradient,
                                                         initial_position=x0,
                                                         max_iterations=n_iter_max)
        error.append(optim_results.objective_value / norm)
        return optim_results.position, error

    elif tl.get_backend() == "jax":
        from jax.scipy.optimize import minimize
        method = 'l-bfgs-experimental-do-not-rely-on-this'
        error = []
        result = minimize(loss, x0, method=method, options={'maxiter': n_iter_max})
        return result.x, error

    elif tl.get_backend() == "mxnet":
        raise ValueError("There is no LBFGS method in Mxnet library")
