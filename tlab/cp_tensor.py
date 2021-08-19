import tensorly as tl
from tensorly.cp_tensor import CPTensor
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_cp_tensor(cp_tensor, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """
    """
    weights, factors = cp_tensor
    if weights is None:
        weights = tl.ones(tl.shape(factors)[1])

    attrs = {"decomposition_type": "CP", "num_modes": len(factors)}
    decomposition_dict = {
        "weights": weights,
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


def load_cp_tensor(path, internal_path):
    attrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "CP")
    weights = decomposition_dict["weights"]
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(attrs["num_modes"])]
    return CPTensor((weights, factors))
