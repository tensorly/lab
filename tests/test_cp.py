from tlab.testing import validate_checkpointing
from tensorly.random import random_cp
from tlab.cp_tensor import store_cp_tensor, load_cp_tensor
from tensorly.testing import assert_array_equal
import pytest


def assert_cp_tensor_equals(cp_tensor1, cp_tensor2):
    if cp_tensor1[0] is None:
        assert_array_equal(cp_tensor2[0], 1)
    else:
        assert_array_equal(cp_tensor2[0], cp_tensor1[0])
    
    for fm1, fm2 in zip(cp_tensor1[1], cp_tensor2[1]):
        assert_array_equal(fm1, fm2)


@pytest.mark.parametrize(
    'shape, num_components, internal_path', 
    [
        ((3, 4, 5), 3, "/"),
        ((3, 4, 5), 6, "/"),
        ((10, 20), 5, "/"),
        ((5, 6, 7, 8), 2, "/"),
        ((3, 4, 5), 3, "/some_path"),
        ((3, 4, 5), 3, "/some_path_nested/some_path"),
    ]
)
def test_cp_checkpoint(shape, num_components, internal_path):
    cp_tensor = random_cp(shape, num_components)
    loaded_cp_tensor = validate_checkpointing(
        cp_tensor, "/", store_cp_tensor, load_cp_tensor
    )
    assert_cp_tensor_equals(cp_tensor, loaded_cp_tensor)

