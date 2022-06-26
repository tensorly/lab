from tempfile import TemporaryDirectory
from pathlib import Path
import h5py
import pytest


def validate_checkpointing(decomposition, internal_path, store_function, load_function):
    """Utility used to validate that checkpoint loading and storing works correctly.

    Parameters:
    -----------
    decomposition : 
        Decomposition object to use for validation.
    internal_path : str
        Internal path to the given HDF5 file that contain the decomposition.
    store_function : Callable
        Function used to store a decomposition to disk.
    load_function : Callable
        Function used to load a decomposition file.

    Returns
    -------
    loaded : 
        Decomposition loaded after storing it to disk.
    """
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        checkpoint = tmpdir / "checkpoint.h5"

        store_function(decomposition, checkpoint, internal_path, compression="gzip",
                       compression_opts=5, fletcher32=True, shuffle=True)

        # Check that loading fails with wrong type
        with h5py.File(checkpoint, "a") as h5:
            group = h5[internal_path]
            decomposition_type = group.attrs["decomposition_type"]
            group.attrs["decomposition_type"] = "TEST_VALUE_SHOULD_FAIL"
        
        with pytest.raises(ValueError):
            load_function(checkpoint, internal_path)
        
        # Reset saved type
        with h5py.File(checkpoint, "a") as h5:
            group = h5[internal_path]
            group.attrs["decomposition_type"] = decomposition_type

        # It should fail if we try to overwrite
        with pytest.raises((RuntimeError, ValueError)):  # RuntimeError for h5py v<2.3, ValueError after
            store_function(
                decomposition,
                checkpoint,
                internal_path,
                compression="gzip",
                compression_opts=5,
                fletcher32=True,
                shuffle=True,
                overwrite=False
            )

        # But not fail if we set overwrite=True
        store_function(
            decomposition,
            checkpoint,
            internal_path,
            compression="gzip",
            compression_opts=5,
            fletcher32=True,
            shuffle=True,
            overwrite=True
        )
        
        loaded = load_function(checkpoint, internal_path)
    return loaded
