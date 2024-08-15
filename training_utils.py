from typing import Dict, Any, Optional, Tuple

import pandas as pd
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image


def create_data_loader(df: pd.DataFrame, config: Dict[str, Any]) -> DataLoader:
    """
    Create a data loader for a dataset.


    Args:
    - images_dir (str): Directory containing the images.
    - df (pd.DataFrame): DataFrame containing the dataset.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data loading.

    Returns:
    - DataLoader: DataLoader for the dataset.
    """
    transform: Optional[Any] = config.get("transform", None)
    batch_size: int = config.get("batch_size", 32)
    num_workers: int = config.get("num_workers", 2)

    dataset = ImageDataset(df, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader


# Model Definition
class SimpleNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(
            784 * 3, 128
        )  # Example for an input size of 784*3 (e.g., flattened 28x28 image) and hidden layer with 128 units
        self.fc2 = nn.Linear(
            128, n_classes
        )  # Output layer with 10 units (e.g., 10 classes for classification)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)  # Flatten the input tensor except for the batch dimension
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: Optional[Any] = None) -> None:
        self.dataframe = dataframe
        self.transform = transform
        self.transform_default = transforms.Compose(
            [
                transforms.Resize([28, 28], antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # replace with your own implementation if needed
        img_path = self.dataframe.iloc[idx]["image_path"]
        image = read_image(img_path).float() / 255.0
        # labels in this dataset start from 1, but we need to start from 0
        label = int(self.dataframe.iloc[idx]["label"]) - 1

        if self.transform:
            image = self.transform(image)
        else:
            image = self.transform_default(image)

        return image, label
