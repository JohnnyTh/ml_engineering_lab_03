import logging

from pathlib import Path
from typing import Tuple, Dict, Any, List, Union

import yaml

import numpy as np
import pandas as pd

import scipy.io as sio

logging.basicConfig(level=logging.INFO)

def train_test_split(
    data: pd.DataFrame, test_size: Union[float, int] = 0.25, random_state: Union[int, None] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters:
    X :  pd.DataFrame
         The input data to split.
    test_size : float, int, or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns:
    Tuple containing:
        - data_train: pd.DataFrame
        - data_test: pd.DataFrame
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(data)
    if isinstance(test_size, float):
        test_size = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    data_train = data.iloc[train_indices]
    data_test = data.iloc[test_indices]

    return data_train, data_test


def load_labels(labels_dir: str) -> pd.DataFrame:
    """
    Load labels from a MAT file.

    Args:
    - labels_dir (str): Path to the labels MAT file dir.

    Returns:
    - pd.DataFrame: DataFrame containing the labels.
    """
    labels_mat = sio.loadmat(f"{labels_dir}/imagelabels.mat")
    labels_df = pd.DataFrame({"label": labels_mat["labels"][0]})

    return labels_df


def find_add_images_to_labels(
    images_dir: str, labels: pd.DataFrame, image_ext: str = "jpg"
) -> pd.DataFrame:
    image_paths = sorted(
        [str(image_path.absolute()) for image_path in Path(images_dir).rglob(f"*.{image_ext}")]
    )
    if len(image_paths) != len(labels):
        logging.error(
            f"Found {len(image_paths)} image_paths but "
            f"{len(labels)} labels were provided, cannot continue."
        )
        raise ValueError

    labels_w_image_paths = labels.copy(deep=True)
    labels_w_image_paths["image_path"] = image_paths
    return labels_w_image_paths


def assign_batches(labels_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Assigns batch labels to a dataframe.

    Args:
    - labels_df (pd.Dataframe): Dataframe containing dataset information in 'label' and 'image_path' columns.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - pd.DataFrame: dataframe with assigned batch labels in 'batch_name' column.
    """
    labels_df_ = labels_df.copy(deep=True)
    labels_df_["batch_name"] = "not_set"

    n_batches = config["n_batches"]
    batch_size = len(labels_df_) // n_batches

    batch_size_current = 0
    for batch_number in range(n_batches):
        if batch_number == (n_batches - 1):
            # select all the remaining data for the last batch
            labels_df_.iloc[batch_size_current:, labels_df_.columns.get_loc("batch_name")] = str(
                batch_number
            )
        else:
            labels_df_.iloc[
                batch_size_current : batch_size_current + batch_size,
                labels_df_.columns.get_loc("batch_name"),
            ] = str(batch_number)

        batch_size_current += batch_size

    return labels_df_


def select_batches(labels_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Selects data from labels_df based on 'batch_names' from config.

    Args:
    - labels_df (pd.Dataframe): .
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - pd.DataFrame:
    """
    batch_names: List[str] = config["batch_names_select"]
    labels_df_ = labels_df.copy(deep=True)

    labels_df_ = labels_df_[labels_df_["batch_name"].isin(batch_names)]

    return labels_df_


def process_data(
    images_dir: str, labels_path: str, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process the data by loading labels, validating data, splitting into train-validation-test sets,
    and optionally saving the splits.

    Args:
    - images_dir (str): Directory containing the images.
    - labels_path (str): Path to the labels MAT file.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and testing DataFrames.
    """
    # Load labels
    labels_df = load_labels(labels_path)

    # Validate data (if needed, adjust based on your specific validation logic)
    valid_labels_df = find_add_images_to_labels(images_dir, labels_df)

    # Split data into train and test
    train_df, test_df = train_test_split(
        valid_labels_df,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
    )

    train_df = assign_batches(train_df, config)
    train_df = select_batches(train_df, config)

    # Further split train_df into train and validation
    train_df, val_df = train_test_split(
        train_df, test_size=config.get("val_size", 0.2), random_state=config.get("random_state", 42)
    )

    logging.info(
        f"Prepared 3 data splits: train, size: {len(train_df)}, val: {len(val_df)}, test: {len(val_df)}"
    )

    return train_df, val_df, test_df


def main():
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    train_df, val_df, test_df = process_data(config["images_dir"], config["labels_dir"], config)
    for save_path, dataframe in (
        (config["train_split_path"], train_df),
        (config["val_split_path"], val_df),
        (config["test_split_path"], test_df),
    ):
        logging.info(f"Dump split to: {save_path}")
        dataframe.to_csv(save_path)


if __name__ == "__main__":
    main()
