import tensorly as tl
from tensorly.tucker_tensor import TuckerTensor
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_tucker_tensor(tucker_tensor, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """
    """
    core, factors = tucker_tensor

    attrs = {"decomposition_type": "Tucker", "num_modes": len(factors)}
    decomposition_dict = {
        "core": core,
        **{f"fm{i}": tl.to_numpy(fm) for i, fm in enumerate(factors)}
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


def load_tucker_tensor(path, internal_path):
    attrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "Tucker")
    core = decomposition_dict["core"]
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(attrs["num_modes"])]
    return TuckerTensor((core, factors))
