from tlab.testing import validate_checkpointing
from tensorly.random import random_tt_matrix
from tlab.tt_matrix import store_tt_matrix, load_tt_matrix
from tensorly.testing import assert_array_equal
import pytest


def assert_tt_matrix_equals(tt_matrix1, tt_matrix2):
    for fm1, fm2 in zip(tt_matrix1, tt_matrix2):
        assert_array_equal(fm1, fm2)


@pytest.mark.parametrize(
    'shape, num_components, internal_path', 
    [
        ((3, 4, 5, 4), 3, "/"),
        ((3, 4, 5, 2), 6, "/"),
        ((10, 20), 5, "/"),
        ((5, 6, 7, 8), (1, 3, 1), "/"),
        ((3, 4, 5, 6, 3, 8), 3, "/some_path"),
        ((3, 4, 5, 9, 3, 5), (1, 3, 2, 1), "/some_path_nested/some_path"),
    ]
)
def test_tt_checkpoint(shape, num_components, internal_path):
    tt_matrix = random_tt_matrix(shape, num_components)
    loaded_tt_matrix = validate_checkpointing(
        tt_matrix, "/", store_tt_matrix, load_tt_matrix
    )
    assert_tt_matrix_equals(tt_matrix, loaded_tt_matrix)
