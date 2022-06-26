from tlab.testing import validate_checkpointing
from tensorly.random import random_parafac2
from tlab.parafac2_tensor import store_parafac2_tensor, load_parafac2_tensor
from tensorly.testing import assert_array_equal
import pytest


def assert_parafac2_tensor_equals(parafac2_tensor1, parafac2_tensor2):
    if parafac2_tensor1[0] is None:
        assert_array_equal(parafac2_tensor2[0], 1)
    else:
        assert_array_equal(parafac2_tensor2[0], parafac2_tensor1[0])
    
    for fm1, fm2 in zip(parafac2_tensor1[1], parafac2_tensor2[1]):
        assert_array_equal(fm1, fm2)
    for p1, p2 in zip(parafac2_tensor1[2], parafac2_tensor2[2]):
        assert_array_equal(p1, p2)


@pytest.mark.parametrize(
    'shape, num_components, internal_path', 
    [
        ([(4, 5), (4, 5), (4, 5)], 3, "/"),
        ([(4, 5), (4, 5), (4, 5)], 4, "/"),
        ([(4, 5), (4, 5), (4, 5)], 3, "/some_path"),
        ([(4, 5), (4, 5), (4, 5)], 3, "/some_path_nested/some_path"),
        ([(4, 5), (9, 5), (7, 5), (6, 5)], 3, "/"),
        ([(4, 5), (9, 5), (7, 5), (6, 5)], 4, "/"),
        ([(4, 5), (9, 5), (7, 5), (6, 5)], 3, "/some_path"),
        ([(4, 5), (9, 5), (7, 5), (6, 5)], 3, "/some_path_nested/some_path"),
    ]
)
def test_parafac2_checkpoint(shape, num_components, internal_path):

    parafac2_tensor = random_parafac2(shape, num_components)
    loaded_parafac2_tensor = validate_checkpointing(
        parafac2_tensor, "/", store_parafac2_tensor, load_parafac2_tensor
    )
    assert_parafac2_tensor_equals(parafac2_tensor, loaded_parafac2_tensor)

