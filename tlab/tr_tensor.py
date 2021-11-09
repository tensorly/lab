import tensorly as tl
from tensorly.tr_tensor import TRTensor
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_tr_tensor(tr_tensor, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """Store a TRTensor to file
    
    Parameters
    ----------
    tr_tensor : TRTensor
        Tensor ring-tensor object to store.
    path : str or pathlib.Path
        Path to the HDF5 file the checkpoint is stored in
    internal_path : str
        Name of the HDF5 group the decomposition is stored in.
    compression : str or None
        Compression type for the HDF5-datasets. Any option supported
        by HDF5 is accepted here.
    overwrite : bool
        If True, then the code will attempt to overwrite existing decompositions.
        However, it may fail (e.g. if the tensor has changed shape).
    **kwargs
        Additional keyword arguments passed to ``h5py.create_dataset``.

    Examples
    --------

    Store a single decomposition in a file.

    >>> tr = random_tr_tensor((10, 15, 20), 3)
    ... store_tr_tensor(tr, "checkpoint_file.h5")
    ... loaded_tr = load_tr_tensor("checkpoint_file.h5")

    Store multiple decompositions in a single file

    >>> tr_1 = random_tr_tensor((10, 15, 20), 3)
    ... tr_2 = random_tr_tensor((10, 15, 20), 5)
    ... store_tr_tensor(tr_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_tr_tensor(tr_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_tr1 = load_tr_tensor("checkpoint_file.h5", "rank3")
    ... loaded_tr2 = load_tr_tensor("checkpoint_file.h5", "rank5")
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
    """Load a TR decomposition from file

    Parameters:
    -----------
    path : str or pathlib.Path
        Path to the HDF5 file the checkpoint is stored in
    internal_path : str
        Name of the HDF5 group the decomposition is stored in.
    

    Examples
    --------

    Store a single decomposition in a file.

    >>> tr = random_tr_tensor((10, 15, 20), 3)
    ... store_tr_tensor(tt, "checkpoint_file.h5")
    ... loaded_tr = load_tr_tensor("checkpoint_file.h5")

    Store multiple decompositions in a single file
    
    >>> tr_1 = random_tr_tensor((10, 15, 20), 3)
    ... tr_2 = random_tr_tensor((10, 15, 20), 5)
    ... store_tr_tensor(tr_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_tr_tensor(tr_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_tt1 = load_tr_tensor("checkpoint_file.h5", "rank3")
    ... loaded_tt2 = load_tr_tensor("checkpoint_file.h5", "rank5")
    """
    atrrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "TRTensor")
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(atrrs["num_modes"])]
    return TRTensor(factors)
