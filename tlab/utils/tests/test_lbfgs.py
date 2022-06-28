import tensorly as tl
from tensorly.testing import assert_array_almost_equal
from .._lbfgs import lbfgs
import numpy as np

def test_lbfgs():
    # LFBGS test can pass following backends, however we will test with only numpy for now
    backends = ["numpy", "jax", "pytorch", "tensorflow"]
    #for i in range(len(backends)):
    for i in range(1):
        tl.set_backend(backends[i])
        a = tl.tensor(np.random.rand(10, 10))
        true_res = tl.tensor(np.random.rand(10, 10))
        b = tl.dot(a, true_res)
        x_init = tl.tensor(np.zeros([tl.shape(true_res)[0], tl.shape(true_res)[1]]))
        loss = lambda x: tl.sum((a @ tl.reshape(x, tl.shape(x_init)) - b)**2)
        result, _ = lbfgs(loss, tl.tensor_to_vec(x_init))
        result = tl.reshape(result, tl.shape(x_init))
        assert_array_almost_equal(true_res, result, decimal=2)
