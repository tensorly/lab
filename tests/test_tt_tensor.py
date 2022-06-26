from tlab.testing import validate_checkpointing
from tensorly.random import random_tt
from tlab.tt_tensor import store_tt_tensor, load_tt_tensor
from tensorly.testing import assert_array_equal
import pytest


def assert_tt_tensor_equals(tt_tensor1, tt_tensor2):
    for fm1, fm2 in zip(tt_tensor1, tt_tensor2):
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
def test_tt_checkpoint(shape, num_components, internal_path):
    tt_tensor = random_tt(shape, num_components)
    loaded_tt_tensor = validate_checkpointing(
        tt_tensor, "/", store_tt_tensor, load_tt_tensor
    )
    assert_tt_tensor_equals(tt_tensor, loaded_tt_tensor)
