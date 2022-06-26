import tensorly as tl
from tensorly.parafac2_tensor import Parafac2Tensor
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_parafac2_tensor(parafac2_tensor, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """Store a PARAFAC2Tensor to file
    
    Parameters
    ----------
    parafac2_tensor : PARAFAC2Tensor
        PARAFAC2-tensor object to store.
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

    >>> parafac2 = random_parafac2_tensor((10, 15, 20), 3)
    ... store_parafac2_tensor(parafac2, "checkpoint_file.h5")
    ... loaded_parafac2 = load_parafac2_tensor("checkpoint_file.h5")

    Store multiple decompositions in a single file

    >>> parafac2_1 = random_parafac2_tensor((10, 15, 20), 3)
    ... parafac2_2 = random_parafac2_tensor((10, 15, 20), 5)
    ... store_parafac2_tensor(parafac2_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_parafac2_tensor(parafac2_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_parafac21 = load_parafac2_tensor("checkpoint_file.h5", "rank3")
    ... loaded_parafac22 = load_parafac2_tensor("checkpoint_file.h5", "rank5")
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
    """Load a PARAFAC2 decomposition from file

    Parameters:
    -----------
    path : str or pathlib.Path
        Path to the HDF5 file the checkpoint is stored in
    internal_path : str
        Name of the HDF5 group the decomposition is stored in.
    

    Examples
    --------

    Store a single decomposition in a file.

    >>> parafac2 = random_parafac2_tensor((10, 15, 20), 3)
    ... store_parafac2_tensor(parafac2, "checkpoint_file.h5")
    ... loaded_parafac2 = load_parafac2_tensor("checkpoint_file.h5")

    Store multiple decompositions in a single file
    
    >>> parafac2_1 = random_parafac2_tensor((10, 15, 20), 3)
    ... parafac2_2 = random_parafac2_tensor((10, 15, 20), 5)
    ... store_parafac2_tensor(parafac2_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_parafac2_tensor(parafac2_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_parafac21 = load_parafac2_tensor("checkpoint_file.h5", "rank3")
    ... loaded_parafac22 = load_parafac2_tensor("checkpoint_file.h5", "rank5")
    """
    attrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "PARAFAC2")
    weights = decomposition_dict["weights"]
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(attrs["num_modes"])]
    projections = [
        tl.tensor(decomposition_dict[f"proj{i}"])
        for i in range(tl.shape(factors[0])[0])
    ]
    return Parafac2Tensor((weights, factors, projections))
