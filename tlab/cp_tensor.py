import tensorly as tl
from tensorly.cp_tensor import CPTensor
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_cp_tensor(cp_tensor, path, internal_path="/", compression="gzip",
                    overwrite=False, **kwargs):
    """
    Parameters
    ----------
    cp_tensor : CPTensor
        CP-tensor object to store.
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

    >>> cp = random_cp_tensor((10, 15, 20), 3)
    ... store_cp_tensor(cp, "checkpoint_file.h5")
    ... loaded_cp = load_cp_tensor("checkpoint_file.h5")

    Store multiple decompositions in a single file

    >>> cp_1 = random_cp_tensor((10, 15, 20), 3)
    ... cp_2 = random_cp_tensor((10, 15, 20), 5)
    ... store_cp_tensor(cp_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_cp_tensor(cp_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_cp1 = load_cp_tensor("checkpoint_file.h5", "rank3")
    ... loaded_cp2 = load_cp_tensor("checkpoint_file.h5", "rank5")
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


def load_cp_tensor(path, internal_path="/"):
    """Load a CP decomposition from file

    Parameters:
    -----------
    path : str or pathlib.Path
        Path to the HDF5 file the checkpoint is stored in
    internal_path : str
        Name of the HDF5 group the decomposition is stored in.
    
    
    Examples
    --------

    Store a single decomposition in a file.

    >>> cp = random_cp_tensor((10, 15, 20), 3)
    ... store_cp_tensor(cp, "checkpoint_file.h5")
    ... loaded_cp = load_cp_tensor("checkpoint_file.h5")

    Store multiple decompositions in a single file
    
    >>> cp_1 = random_cp_tensor((10, 15, 20), 3)
    ... cp_2 = random_cp_tensor((10, 15, 20), 5)
    ... store_cp_tensor(cp_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_cp_tensor(cp_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_cp1 = load_cp_tensor("checkpoint_file.h5", "rank3")
    ... loaded_cp2 = load_cp_tensor("checkpoint_file.h5", "rank5")
    """
    attrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "CP")
    weights = decomposition_dict["weights"]
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(attrs["num_modes"])]
    return CPTensor((weights, factors))
