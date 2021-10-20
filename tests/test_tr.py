from tlab.testing import validate_checkpointing
from tensorly.random import random_tr
from tlab.tr_tensor import store_tr_tensor, load_tr_tensor
from tensorly.testing import assert_array_equal
import pytest


def assert_tr_tensor_equals(tr_tensor1, tr_tensor2):
    for fm1, fm2 in zip(tr_tensor1, tr_tensor2):
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
def test_tr_checkpoint(shape, num_components, internal_path):
    tr_tensor = random_tr(shape, num_components)
    loaded_tr_tensor = validate_checkpointing(
        tr_tensor, "/", store_tr_tensor, load_tr_tensor
    )
    assert_tr_tensor_equals(tr_tensor, loaded_tr_tensor)
