import tensorly as tl
from tensorly.tr_tensor import TRTensor
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_tr_tensor(tr_tensor, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """
    """
    attrs = {"decomposition_type": "TRTensor", "num_modes": len(tr_tensor)}
    decomposition_dict = {
        **{f"fm{i}": tl.to_numpy(fm) for i, fm in enumerate(tr_tensor)}
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


def load_tr_tensor(path, internal_path):
    atrrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "TRTensor")
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(atrrs["num_modes"])]
    return TRTensor(factors)
