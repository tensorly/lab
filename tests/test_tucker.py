from tlab.testing import validate_checkpointing
from tensorly.random import random_tucker
from tlab.tucker_tensor import store_tucker_tensor, load_tucker_tensor
from tensorly.testing import assert_array_equal
import pytest


def assert_tucker_tensor_equals(tucker_tensor1, tucker_tensor2):
    assert_array_equal(tucker_tensor2[0], tucker_tensor1[0])
    
    for fm1, fm2 in zip(tucker_tensor1[1], tucker_tensor2[1]):
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
def test_tucker_checkpoint(shape, num_components, internal_path):
    tucker_tensor = random_tucker(shape, num_components)
    loaded_tucker_tensor = validate_checkpointing(
        tucker_tensor, "/", store_tucker_tensor, load_tucker_tensor
    )
    assert_tucker_tensor_equals(tucker_tensor, loaded_tucker_tensor)

