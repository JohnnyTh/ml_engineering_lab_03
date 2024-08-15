import logging
import yaml

import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from training_utils import create_data_loader, SimpleNN

logging.basicConfig(level=logging.INFO)


def test_model(
    model: nn.Module, test_loader: DataLoader, loss_function: nn.Module, device: torch.device
) -> float:
    """
    Test a trained model on a test dataset and compute test metrics.


    Args:
    - model (nn.Module): Trained PyTorch model.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - loss_function (nn.Module): Loss function for computing the loss.
    - device (torch.device): Device (CPU or GPU) on which to run the evaluation.

    Returns:
    - float: Test loss.
    """

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    logging.info(f"Test Loss: {test_loss:.4f}")

    return test_loss


def main() -> None:
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    test_df = pd.read_csv(config["val_split_path"], index_col=0)
    test_loader = create_data_loader(test_df, config)

    model = SimpleNN(n_classes=102).to(device_)
    loss_function = nn.CrossEntropyLoss()

    test_model(model, test_loader, loss_function, device_)


if __name__ == "__main__":
    main()
