from tempfile import TemporaryDirectory
from pathlib import Path
import h5py
import pytest


def validate_checkpointing(decomposition, internal_path, store_function, load_function):
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        checkpoint = tmpdir / "checkpoint.h5"

        store_function(decomposition, checkpoint, internal_path, compression="gzip",
                       compression_opts=5, fletcher32=True, shuffle=True)
        with h5py.File(checkpoint, "a") as h5:
            group = h5[internal_path]
            decomposition_type = group.attrs["decomposition_type"]
            group.attrs["decomposition_type"] = "TEST_VALUE_SHOULD_FAIL"
        
        with pytest.raises(ValueError):
            load_function(checkpoint, internal_path)
        
        with h5py.File(checkpoint, "a") as h5:
            group = h5[internal_path]
            group.attrs["decomposition_type"] = decomposition_type

        with pytest.raises(RuntimeError):
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
