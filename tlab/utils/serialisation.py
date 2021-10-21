from pathlib import Path
import tensorly as tl

import h5py


def _store_decomposition_dict(decomposition_dict, path, internal_path, attrs,
                              compression="gzip", overwrite=False, **kwargs):
    """Utility function to store a decomposition to an HDF5 file

    Parameters:
    -----------
    decomposition_dict : Dict[str, np.ndarray]
    path : str or pathlib.Path
        Path to the HDF5 file the checkpoint is stored in
    internal_path : str
        Name of the HDF5 group the decomposition is stored in.
    
    decomposition_type : str
        Name of the decomposition type, used for validation when the decomposition
        is loaded.
    """
    if "decomposition_type" not in attrs:
        raise ValueError("`'decomposition_type'` must be a key in `attrs`.",
                         "\nThis error message should only occur during ",
                         "development of tensorly. If you're not currently "
                         "modifying the TensorLy source code, then something "
                         "has gone seriously wrong!")

    if overwrite:
        get_dataset = 'require_dataset'
        get_group = 'require_group'
    else:
        get_dataset = 'create_dataset'
        get_group = 'create_group'
    with h5py.File(path, "a") as h5:
        if internal_path in {"/", ""}:
            group = h5
        else:
            group = getattr(h5, get_group)(internal_path)

        i = 0
        for name, data in decomposition_dict.items():
            i += 1
            getattr(group, get_dataset)(
                name, shape=tl.shape(data), dtype=data.dtype, data=data,
                compression=compression, **kwargs
            )

        
        for attr, value in attrs.items():
            group.attrs[attr] = value

def _load_decomposition_dict(path, internal_path, decomposition_type):
    """Utility function to load a decomposition from an HDF5 file

    Parameters:
    -----------
    path : str or pathlib.Path
        Path to the HDF5 file the checkpoint is stored in
    internal_path : str
        Name of the HDF5 group the decomposition is stored in.
    decomposition_type : str
        Name of the decomposition type, used to validate that the checkpoint
        is loaded correctly.
    """
    with h5py.File(path, "r") as h5:
        group = h5[internal_path]

        # Validate checkpoint
        if "decomposition_type" not in group.attrs:
            raise ValueError("Internal path does not point to a valid TensorLy decomposition")
        
        file_type = group.attrs["decomposition_type"]
        if file_type != decomposition_type:
            raise ValueError(f"Checkpoint is a {file_type} tensor, not a {decomposition_type} tensor.")

        decomposition_dict = {key: value[:] for key, value in group.items()}
        attrs = dict(group.attrs)
    
    return attrs, decomposition_dict