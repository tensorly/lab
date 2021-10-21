import tensorly as tl
from tensorly.tt_matrix import TTMatrix
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_tt_matrix(tt_matrix, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """
    """
    attrs = {"decomposition_type": "TTMatrix", "num_modes": len(tt_matrix)}
    decomposition_dict = {
        **{f"fm{i}": tl.to_numpy(fm) for i, fm in enumerate(tt_matrix)}
    }

    return _store_decomposition_dict(
        decomposition_dict=decomposition_dict,
        path=path,
        internal_path=internal_path,
        attrs=attrs,
        compression=compression,
        overwrite=overwrite,
        **kwargs
    )


def load_tt_matrix(path, internal_path):
    attrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "TTMatrix")
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(attrs["num_modes"])]
    return TTMatrix(factors)
