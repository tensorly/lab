import tensorly as tl
from tensorly.parafac2_tensor import Parafac2Tensor
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_parafac2_tensor(parafac2_tensor, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """
    """
    weights, factors, projections = parafac2_tensor
    if weights is None:
        weights = tl.ones(tl.shape(factors)[1])

    attrs = {"decomposition_type": "PARAFAC2", "num_modes": len(factors)}
    decomposition_dict = {
        "weights": weights,
        **{f"fm{i}": tl.to_numpy(fm) for i, fm in enumerate(factors)},
        **{f"proj{i}": tl.to_numpy(proj) for i, proj in enumerate(projections)}
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


def load_parafac2_tensor(path, internal_path):
    attrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "PARAFAC2")
    weights = decomposition_dict["weights"]
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(attrs["num_modes"])]
    projections = [
        tl.tensor(decomposition_dict[f"proj{i}"])
        for i in range(tl.shape(factors[0])[0])
    ]
    return Parafac2Tensor((weights, factors, projections))
