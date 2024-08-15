import logging
import yaml

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)


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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: Path = Path("best_model.pth"),
) -> Path:
    model.to(device)
    best_val_loss: float = float("inf")
    best_model_path: Path = Path()

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            # Forward pass
            outputs: torch.Tensor = model(batch_inputs)
            loss: torch.Tensor = loss_function(outputs, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        # Validation step
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            val_loss: float = 0.0
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs: torch.Tensor = model(val_inputs)
                val_loss += loss_function(val_outputs, val_targets).item()

            val_loss /= len(val_loader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = save_path
                torch.save(model.state_dict(), best_model_path)

                logging.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

    logging.info("Training complete.")

    return best_model_path


def main():
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    train_df = pd.read_csv(config["train_split_path"], index_col=0)
    val_df = pd.read_csv(config["val_split_path"], index_col=0)

    train_loader = create_data_loader(train_df, config)
    val_loader = create_data_loader(val_df, config)

    model = SimpleNN(n_classes=102).to(device_)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    train_model(
        model,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        num_epochs=10,
        device=device_,
        save_path=Path(config["model_save_path"]),
    )

if __name__ == '__main__':
    main()