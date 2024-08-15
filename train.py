import logging
import yaml

from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from training_utils import create_data_loader, SimpleNN

logging.basicConfig(level=logging.INFO)


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


def main() -> None:
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