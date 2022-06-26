import tensorly as tl
from tensorly.tucker_tensor import TuckerTensor
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_tucker_tensor(tucker_tensor, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """Store a TuckerTensor to file
    
    Parameters
    ----------
    tucker_tensor : TuckerTensor
        Tucker-tensor object to store.
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

    >>> tucker = random_tucker_tensor((10, 15, 20), 3)
    ... store_tucker_tensor(tucker, "checkpoint_file.h5")
    ... loaded_tucker = load_tucker_tensor("checkpoint_file.h5")

    Store multiple decompositions in a single file

    >>> tucker_1 = random_tucker_tensor((10, 15, 20), 3)
    ... tucker_2 = random_tucker_tensor((10, 15, 20), 5)
    ... store_tucker_tensor(tucker_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_tucker_tensor(tucker_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_tucker1 = load_tucker_tensor("checkpoint_file.h5", "rank3")
    ... loaded_tucker2 = load_tucker_tensor("checkpoint_file.h5", "rank5")
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
    """Load a Tucker decomposition from file

    Parameters:
    -----------
    path : str or pathlib.Path
        Path to the HDF5 file the checkpoint is stored in
    internal_path : str
        Name of the HDF5 group the decomposition is stored in.
    

    Examples
    --------

    Store a single decomposition in a file.

    >>> tucker = random_tucker_tensor((10, 15, 20), 3)
    ... store_tucker_tensor(tucker, "checkpoint_file.h5")
    ... loaded_tucker = load_tucker_tensor("checkpoint_file.h5")

    Store multiple decompositions in a single file
    
    >>> tucker_1 = random_tucker_tensor((10, 15, 20), 3)
    ... tucker_2 = random_tucker_tensor((10, 15, 20), 5)
    ... store_tucker_tensor(tucker_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_tucker_tensor(tucker_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_tucker1 = load_tucker_tensor("checkpoint_file.h5", "rank3")
    ... loaded_tucker2 = load_tucker_tensor("checkpoint_file.h5", "rank5")
    """
    attrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "Tucker")
    core = decomposition_dict["core"]
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(attrs["num_modes"])]
    return TuckerTensor((core, factors))
