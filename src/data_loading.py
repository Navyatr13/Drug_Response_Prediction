# src/data_loading.py
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_h5_data(file_path, dataset_name):
    """
    Load data from a generic .h5 file.

    Args:
        file_path: Path to the .h5 file.
        dataset_name: The name of the dataset within the file.

    Returns:
        DataFrame containing the dataset.
    """
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return pd.DataFrame(data)


def normalize_data(data):
    """Normalize omics data."""
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


def preprocess_h5_data(file_path, dataset_name):
    """
    Load and preprocess data from a generic .h5 file.

    Args:
        file_path: Path to the .h5 file.
        dataset_name: Dataset name within the file.

    Returns:
        Normalized DataFrame.
    """
    raw_data = load_h5_data(file_path, dataset_name)
    normalized_data = normalize_data(raw_data)
    return normalized_data
