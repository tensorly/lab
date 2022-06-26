import tensorly as tl
from tensorly.tt_matrix import TTMatrix
from .utils.serialisation import _store_decomposition_dict, _load_decomposition_dict


def store_tt_matrix(tt_matrix, path, internal_path, compression="gzip",
                    overwrite=False, **kwargs):
    """Store a TTMatrix to file
    
    Parameters
    ----------
    tt_matrix : TTMatrix
        TT-matrix object to store.
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

    >>> tt = random_tt_matrix((10, 15, 20), 3)
    ... store_tt_matrix(tt, "checkpoint_file.h5")
    ... loaded_tt = load_tt_matrix("checkpoint_file.h5")

    Store multiple decompositions in a single file

    >>> tt_1 = random_tt_matrix((10, 15, 20), 3)
    ... tt_2 = random_tt_matrix((10, 15, 20), 5)
    ... store_tt_matrix(tt_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_tt_matrix(tt_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_tt1 = load_tt_matrix("checkpoint_file.h5", "rank3")
    ... loaded_tt2 = load_tt_matrix("checkpoint_file.h5", "rank5")
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
    """Load a TT matrix decomposition from file

    Parameters:
    -----------
    path : str or pathlib.Path
        Path to the HDF5 file the checkpoint is stored in
    internal_path : str
        Name of the HDF5 group the decomposition is stored in.
    

    Examples
    --------

    Store a single decomposition in a file.

    >>> tt = random_tt_matrix((10, 15, 20), 3)
    ... store_tt_matrix(tt, "checkpoint_file.h5")
    ... loaded_tt = load_tt_matrix("checkpoint_file.h5")

    Store multiple decompositions in a single file
    
    >>> tt_1 = random_tt_matrix((10, 15, 20), 3)
    ... tt_2 = random_tt_matrix((10, 15, 20), 5)
    ... store_tt_matrix(tt_1, "checkpoint_file.h5", internal_path="rank3")
    ... store_tt_matrix(tt_2, "checkpoint_file.h5", internal_path="rank5")
    ... loaded_tt1 = load_tt_matrix("checkpoint_file.h5", "rank3")
    ... loaded_tt2 = load_tt_matrix("checkpoint_file.h5", "rank5")
    """
    attrs, decomposition_dict = _load_decomposition_dict(path, internal_path, "TTMatrix")
    factors = [tl.tensor(decomposition_dict[f"fm{i}"]) for i in range(attrs["num_modes"])]
    return TTMatrix(factors)
