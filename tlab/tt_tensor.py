import tensorly as tl
from tensorly.tt_tensor import TTTensor
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_tt_tensor(tt_tensor, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """
    """
    attrs = {"decomposition_type": "TTTensor", "num_modes": len(tt_tensor)}
    decomposition_dict = {
        **{f"fm{i}": tl.to_numpy(fm) for i, fm in enumerate(tt_tensor)}
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


def load_tt_tensor(path, internal_path):
    attrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "TTTensor")
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(attrs["num_modes"])]
    return TTTensor(factors)
