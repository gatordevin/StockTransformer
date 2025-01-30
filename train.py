import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Union, Dict
from sklearn.preprocessing import StandardScaler
import pickle
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from stockdata import StockDataProcessor, StockDatasetGenerator, StockData
from model import TransformerModel
from data import StockDataset
from data import custom_collate_fn
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        feature_names: Dict[str, List[str]],
        output_target: str,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        weight_decay: float = 1e-5,
        patience: int = 10,  # For early stopping
        save_path: str = 'best_model.pth'
    ):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_dataset (torch.utils.data.Dataset): Training dataset.
            valid_dataset (torch.utils.data.Dataset): Validation dataset.
            test_dataset (torch.utils.data.Dataset): Test dataset.
            feature_names (Dict[str, List[str]]): Dictionary mapping symbols to their feature names.
            output_target (str): The target variable to predict.
            device (torch.device): Device to train on.
            batch_size (int): Batch size for DataLoader.
            learning_rate (float): Learning rate for the optimizer.
            num_epochs (int): Maximum number of training epochs.
            weight_decay (float): Weight decay (L2 penalty) for the optimizer.
            patience (int): Number of epochs to wait for improvement before early stopping.
            save_path (str): Path to save the best model.
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.feature_names = feature_names
        self.output_target = output_target
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.save_path = save_path

        # Initialize DataLoaders with custom_collate
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, 
            drop_last=False, collate_fn=custom_collate_fn
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False, 
            drop_last=False, collate_fn=custom_collate_fn
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, 
            drop_last=False, collate_fn=custom_collate_fn
        )

        # Define Loss Function
        self.criterion = self.gaussian_nll_loss

        # Define Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Define Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )

        # Early Stopping Parameters
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        # Logging
        self.logger = logging.getLogger('Trainer')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def gaussian_nll_loss(self, output, target):
        """
        Negative Log-Likelihood loss for Gaussian distribution.

        Args:
            output (torch.Tensor): Model output of shape (batch_size, 2) where the second column is variance.
            target (torch.Tensor): Ground truth targets of shape (batch_size,).

        Returns:
            torch.Tensor: Computed NLL loss.
        """
        mean = output[:, 0]
        var = output[:, 1]
        nll = 0.5 * torch.log(2 * torch.pi * var) + ((target - mean) ** 2) / (2 * var)
        return torch.mean(nll)

    def train_one_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            inputs = batch['input'].to(self.device)  # Shape: (batch_size, seq_len, num_features)
            targets = batch['target'].to(self.device)  # Shape: (batch_size,)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)  # Shape: (batch_size, 2)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def evaluate(self, loader: DataLoader):
        """
        Evaluate the model on a given dataset.

        Args:
            loader (DataLoader): DataLoader for the dataset to evaluate.

        Returns:
            Tuple[float, float, float]: Tuple containing average loss, MAE, and MSE.
        """
        self.model.eval()
        running_loss = 0.0
        mae = 0.0
        mse = 0.0

        with torch.no_grad():
            for batch in loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                outputs = self.model(inputs)  # Shape: (batch_size, 2)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

                # Compute MAE and MSE
                preds = outputs[:, 0]
                mae += torch.sum(torch.abs(preds - targets)).item()
                mse += torch.sum((preds - targets) ** 2).item()

        avg_loss = running_loss / len(loader.dataset)
        avg_mae = mae / len(loader.dataset)
        avg_mse = mse / len(loader.dataset)
        return avg_loss, avg_mae, avg_mse

    def fit(self):
        """
        Execute the training loop over multiple epochs.
        """
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss, val_mae, val_mse = self.evaluate(self.valid_loader)

            self.scheduler.step(val_loss)

            self.logger.info(
                f"Epoch [{epoch}/{self.num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val MAE: {val_mae:.4f} "
                f"Val MSE: {val_mse:.4f}"
            )

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                # Save the best model
                torch.save(self.model.state_dict(), self.save_path)
                self.logger.info(f"Validation loss improved. Model saved to {self.save_path}.")
            else:
                self.epochs_no_improve += 1
                self.logger.info(f"No improvement in validation loss for {self.epochs_no_improve} epochs.")

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                self.logger.info("Early stopping triggered.")
                break

    def load_best_model(self):
        """
        Load the best model saved during training.
        """
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
            self.logger.info(f"Loaded best model from {self.save_path}.")
        else:
            self.logger.warning(f"No model found at {self.save_path}.")

    def test(self):
        """
        Evaluate the model on the test dataset.

        Returns:
            Tuple[float, float, float]: Tuple containing test loss, MAE, and MSE.
        """
        self.load_best_model()
        test_loss, test_mae, test_mse = self.evaluate(self.test_loader)
        self.logger.info(
            f"Test Loss: {test_loss:.4f} "
            f"Test MAE: {test_mae:.4f} "
            f"Test MSE: {test_mse:.4f}"
        )
        return test_loss, test_mae, test_mse
    
if __name__ == '__main__':
    # -------------------------------
    # Configure Logging
    # -------------------------------
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # -------------------------------
    # Dataset Instantiation
    # -------------------------------
    API_KEY = os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_API_SECRET")
    BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    stock_data = StockData(API_KEY, API_SECRET, BASE_URL)

    # Initialize StockDataProcessor
    processor = StockDataProcessor(stock_data)

    # Create an instance of the StockDatasetGenerator with default parameters.
    stock_dataset_generator = StockDatasetGenerator(
        stock_data_processor=processor,
        input_sequence_length=24,
        prediction_horizon=12,
        dataset_save_path='data/stock_dataset.pkl',  # Specify directory
        scalers_save_path='data/scalers.pkl',
        train_split=0.8,
        validation_split=0.1,
        test_split=0.1
    )
    stock_dataset_generator.load_dataset()

    # -------------------------------
    # Create PyTorch Datasets for Each Split
    # -------------------------------
    target = 'total_log_return'  # Ensure consistency with StockDatasetGenerator's output_target

    train_dataset = StockDataset(stock_dataset_generator, mode='train', output_target=target)
    valid_dataset = StockDataset(stock_dataset_generator, mode='valid', output_target=target)
    test_dataset = StockDataset(stock_dataset_generator, mode='test', output_target=target)

    # -------------------------------
    # Initialize and Configure the Model
    # -------------------------------
    input_dim = len(train_dataset.feature_names.get(train_dataset.data[0][2], []))  # Assuming feature_names mapped by symbol
    if input_dim == 0:
        # Fallback if feature_names are not properly mapped
        input_dim = len(train_dataset.feature_names[next(iter(train_dataset.feature_names))])

    model = TransformerModel(input_dim=input_dim)

    # -------------------------------
    # Initialize the Trainer
    # -------------------------------
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        feature_names=train_dataset.feature_names,  # Dictionary mapping symbols to feature lists
        output_target=target,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=100,
        weight_decay=1e-5,
        patience=10,
        save_path='best_transformer_model.pth'
    )

    # -------------------------------
    # Start Training
    # -------------------------------
    trainer.fit()

    # -------------------------------
    # Evaluate on Test Set
    # -------------------------------
    trainer.test()