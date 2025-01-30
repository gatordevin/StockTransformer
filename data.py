import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Union, Dict
from sklearn.preprocessing import StandardScaler
import pickle
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


from stockdata import StockDataProcessor, StockDatasetGenerator, StockData  # Ensure these are correctly defined/imported

def custom_collate_fn(batch):
    """
    Custom collate function to handle pandas.Series and strings.
    Converts 'label' to dictionaries and keeps 'symbol' as a list of strings.
    
    Args:
        batch (list): A list of data samples, each a dictionary.
        
    Returns:
        dict: A dictionary with batched 'input', 'target', 'label', and 'symbol'.
    """
    # Stack 'input' tensors
    inputs = torch.stack([item['input'] for item in batch], dim=0)  # Shape: (batch_size, seq_len, num_features)
    
    # Stack 'target' tensors
    targets = torch.stack([item['target'] for item in batch], dim=0)  # Shape: (batch_size,)
    
    # Convert 'label' (pandas.Series) to a list of dictionaries
    labels = [item['label'].to_dict() for item in batch]  # List of dicts
    
    # Collect 'symbol' as a list of strings
    symbols = [item['symbol'] for item in batch]  # List of strings
    
    return {
        'input': inputs,
        'target': targets,
        'label': labels,
        'symbol': symbols
    }

class StockDataset(Dataset):
    def __init__(self, stock_dataset_generator, mode='train', output_target='total_log_return'):
        """
        Initialize the StockDataset.

        Args:
            stock_dataset_generator: Instance of StockDatasetGenerator.
            mode: One of 'train', 'valid', or 'test' to specify the dataset split.
            output_target: The target feature to predict.
        """
        # Load the datasets and scalers
        self.output_target = output_target
        # Retrieve the appropriate dataset split
        if mode == 'train':
            self.data = stock_dataset_generator.get_train_dataset()
        elif mode == 'valid':
            self.data = stock_dataset_generator.get_validation_dataset()
        elif mode == 'test':
            self.data = stock_dataset_generator.get_test_dataset()
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'valid', or 'test'.")

        # Shuffle the data to eliminate temporal dependencies
        random.shuffle(self.data)

        # Extract scalers from the generator
        self.scalers = stock_dataset_generator.scalers
        self.feature_names = stock_dataset_generator.feature_names
        self.mode = mode

    def __len__(self):
        """
        Return the number of samples in the dataset split.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the input sequence and label for the given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - 'input': Tensor of input features.
                - 'target': Tensor of the target value.
                - 'label': Original label as a pandas Series.
                - 'symbol': Stock symbol.
        """
        input_sequence, label, symbol = self.data[idx]

        # Convert input_sequence (pd.DataFrame) to tensor
        # Assuming input_sequence is a DataFrame, convert it to a NumPy array first
        input_array = input_sequence.to_numpy(dtype=np.float32)
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # Convert label (pd.Series) to tensor
        label_value = label[self.output_target]
        label_tensor = torch.tensor(label_value, dtype=torch.float32)

        return {'input': input_tensor, 'target': label_tensor, 'label': label, 'symbol': symbol}

    def get_scaler(self, symbol):
        """
        Retrieve the scaler for a specific stock symbol.

        Args:
            symbol: Stock symbol (string).

        Returns:
            The StandardScaler instance for the given symbol.
        """
        return self.scalers.get(symbol.upper())


def compute_statistics(dataset: StockDataset, feature_names: List[str]):
    """
    Compute and print statistical summaries for the dataset.

    Args:
        dataset: Instance of StockDataset.
        feature_names: List of feature names associated with the dataset's symbol.
    """
    inputs = []
    targets = []

    for sample in dataset:
        inputs.append(sample['input'].numpy())
        targets.append(sample['target'].item())

    # Convert inputs to a single NumPy array: [num_samples, input_length, num_features]
    inputs = np.array(inputs)  # Shape: (num_samples, input_length, num_features)
    targets = np.array(targets)  # Shape: (num_samples,)

    # Reshape inputs to [num_samples * input_length, num_features] for statistical computation
    num_samples, input_length, num_features = inputs.shape
    inputs_reshaped = inputs.reshape(-1, num_features)  # Shape: (num_samples * input_length, num_features)

    # Create a DataFrame for easier computation
    df_inputs = pd.DataFrame(inputs_reshaped, columns=feature_names)
    df_targets = pd.Series(targets, name='target')

    # Compute statistics for inputs
    input_stats = df_inputs.describe().T  # Transpose for readability

    # Compute statistics for targets
    target_stats = df_targets.describe()

    print("\n===== Dataset Statistics =====")
    print(f"Mode: {dataset.mode.capitalize()}")
    print(f"Number of Samples: {len(dataset)}\n")

    # Print keys present in each data sample
    if len(dataset) > 0:
        sample_keys = dataset[0].keys()
        print(f"Keys in each data sample: {list(sample_keys)}\n")

    # Print shapes
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample 'input' tensor shape: {sample['input'].shape}")  # Expected: [input_length, num_features]
        print(f"Sample 'target' tensor shape: {sample['target'].shape}\n")  # Expected: torch.Size([]) or torch.Size([1])

    # Print feature names
    print("Feature Names:")
    print(feature_names)
    print("\n")

    # Print input statistics
    print("Input Features Statistics:")
    print(input_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    print("\n")

    # Print target statistics
    print("Target Variable Statistics:")
    print(target_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    print("\n")
    print("===== End of Statistics =====\n")


if __name__ == '__main__':
    # Configure logging
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
    # Print Dataset Details
    # -------------------------------
    for dataset_split in [train_dataset, valid_dataset, test_dataset]:
        if len(dataset_split) == 0:
            logging.warning(f"The {dataset_split.mode} dataset is empty.")
            continue

        # Retrieve feature names for the first sample's symbol
        first_sample = dataset_split[0]
        symbol = first_sample['symbol']
        feature_names = dataset_split.feature_names.get(symbol.upper(), [])

        print(f"===== {dataset_split.mode.capitalize()} Dataset =====")
        print(f"Number of Samples: {len(dataset_split)}")
        print(f"Keys in each data sample: {list(first_sample.keys())}")
        print(f"Sample 'input' tensor shape: {first_sample['input'].shape}")
        print(f"Sample 'target' tensor shape: {first_sample['target'].shape}\n")

        # Compute and print statistics
        compute_statistics(dataset_split, feature_names)