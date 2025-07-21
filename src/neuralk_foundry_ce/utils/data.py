import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
from pathlib import Path


def array_to_dataframe(array: np.ndarray, prefix: str = "column") -> pd.DataFrame:
    """
    Convert a NumPy array to a Pandas DataFrame with prefixed column names.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (n_samples, n_features).
    prefix : str
        Prefix to use for column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with column names like prefix_1, prefix_2, ...
    """
    n_cols = array.shape[1]
    column_names = [f"{prefix}_{i+1}" for i in range(n_cols)]
    return pd.DataFrame(array, columns=column_names)


def make_deduplication(num_samples=10000, embed_dim=128, dup_frac=0.7, avg_dups=3.0, decay=1.0):
    """
    Generates a synthetic dataset for record linkage with embedded duplicates.
    
    Parameters
    ----------
    num_samples : int
        Total number of unique samples to generate before duplication.
    
    embed_dim : int
        Dimensionality of the embedding space for each sample.
    
    dup_frac : float
        Fraction of the final dataset that will consist of duplicates.
        Must be between 0 and 1.
    
    avg_dups : float
        Average number of duplicates per duplicated sample.
        Only applies to samples selected for duplication.
    
    decay : float
        Exponent controlling the steepness of the duplication count distribution.
        Higher values lead to fewer duplicates per sample on average (power-law decay).
    
    Returns
    -------
    df : pd.DataFrame (n_total_samples, embed_dim + 1)
        The full dataset containing original, duplicated samples, and targets.
    
    target_col : str
        Name of the target column in the DataFrame
    """

    total_dups = int(num_samples * dup_frac)
    num_unique = num_samples - total_dups
    num_duplicated_ids = int(total_dups / avg_dups)

    # Generate unique IDs
    next_id = 0
    unique_ids = np.arange(next_id, next_id + num_unique - num_duplicated_ids)
    next_id += len(unique_ids)

    # Generate IDs to duplicate using decreasing function
    weights = 1.0 / (np.arange(1, num_duplicated_ids + 1) ** decay)
    weights /= weights.sum()

    counts = np.floor(weights * total_dups).astype(int)
    # Ensure the total is exactly total_dups
    while counts.sum() < total_dups:
        counts[np.argmin(counts)] += 1
    while counts.sum() > total_dups:
        counts[np.argmax(counts)] -= 1

    duplicated_ids = np.arange(next_id, next_id + num_duplicated_ids)
    next_id += num_duplicated_ids

    duplicated_id_list = np.repeat(duplicated_ids, counts)
    base_id_list = np.concatenate([unique_ids, duplicated_ids])
    all_ids = np.concatenate([base_id_list, duplicated_id_list])
    np.random.shuffle(all_ids)

    embeddings = np.random.randn(num_samples, embed_dim) + (all_ids[:, None] % 50) * 0.05
    embeddings = pd.DataFrame(embeddings, columns=[str(i) for i in range(embed_dim)])
    embeddings.insert(0, 'group_id', all_ids)

    return embeddings, 'group_id'


def load_dataframe(filepath, **kwargs):
    """
    Load a DataFrame from a file. Supports:
    - .csv
    - .tsv
    - .json
    - .parquet
    - .xlsx
    - .xls
    - .feather
    - .pkl / .pickle
    - .h5 (HDF5)
    
    Additional keyword arguments are passed to the pandas reader.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif ext == '.tsv':
        return pd.read_csv(filepath, sep='\t', **kwargs)
    elif ext == '.json':
        return pd.read_json(filepath, **kwargs)
    elif ext == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(filepath, **kwargs)
    elif ext == '.feather':
        return pd.read_feather(filepath, **kwargs)
    elif ext in ['.pkl', '.pickle']:
        return pd.read_pickle(filepath, **kwargs)
    elif ext == '.h5':
        return pd.read_hdf(filepath, **kwargs)
    elif ext == '.npy':
        return np.load(filepath, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    

def get_dataset_dir() -> Path:
    """
    Get the local path to the dataset directory.

    Returns
    -------
    path : Path
        Path to the dataset directory.
    """
    base_dir = (
        Path(os.environ.get("NEURALK_DATASETS_DIR"))
        if "NEURALK_DATASETS_DIR" in os.environ
        else Path.home() / ".neuralk" / "datasets"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def load_json(path: Path):
    """
    Load a JSON file if it exists, otherwise return None.

    Parameters:
    - path: Path to the JSON file

    Returns:
    - The loaded JSON object or None if the file does not exist
    """
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def dump_json(path: Path, obj, indent=2):
    """
    Dump a Python object to a JSON file.

    Parameters:
    - obj: The Python object to serialize
    - path: Output file path
    - indent: Indentation level (default: 2)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
