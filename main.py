import math
import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 1) Helper functions for data
# -----------------------------

def calculate_bollinger_bands(data, window=10, num_of_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band - lower_band

def calculate_rsi(data, window=10):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(data, periods=10):
    return ((data - data.shift(periods)) / data.shift(periods)) * 100

def calculate_stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    """
    Returns %K and %D of the Stochastic Oscillator.
    %K = 100 * (close - minLow) / (maxHigh - minLow) over 'k_period'
    %D = rolling mean of %K over 'd_period'
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_line = k_line.rolling(window=d_period).mean()
    return k_line, d_line

def get_ticker_data(ticker, period="1mo", interval="5m"):
    """
    Check if CSV for this ticker/period/interval exists.
    If not, download from yfinance and save.
    """
    filename = f"{ticker}_{period}_{interval}.csv"
    
    if os.path.exists(filename):
        # Load existing data
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        # Download data
        df = yf.download(ticker, period=period, interval=interval)
        if not df.empty:
            df.to_csv(filename)
    return df

# -----------------------------
# 2) Download/preprocess data
# -----------------------------

tickers = ['META', 'AAPL', 'MSFT', 'AMZN', 'GOOG']
sequence_length = 24  # 2 hours of 5-minute intervals
future_gap = 12       # Number of steps to look ahead
stats = {}

ticker_data_frames = []
for ticker in tickers:
    data = get_ticker_data(ticker, period="1mo", interval="5m")
    
    # Check if the data is valid
    if data.empty or len(data) < 20:  # Minimal check
        print(f"[WARNING] Not enough data for {ticker} or data is empty. Skipping.")
        continue

    # Extract columns
    close = data['Close'].squeeze()  
    volume = data['Volume'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()

    # Calculate indicators
    width = calculate_bollinger_bands(close, window=14)
    rsi = calculate_rsi(close, window=14)
    roc = calculate_roc(close, periods=14)
    diff = close.diff(1)
    pct_change = close.pct_change() * 100

    # Calculate Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic_oscillator(high, low, close, k_period=14, d_period=3)

    # Create the DataFrame
    ticker_df = pd.DataFrame({
        f'{ticker}_close': close,
        f'{ticker}_width': width,
        f'{ticker}_rsi': rsi,
        f'{ticker}_roc': roc,
        f'{ticker}_volume': volume,
        f'{ticker}_diff': diff,
        f'{ticker}_pct_change': pct_change,
        f'{ticker}_stoch_k': stoch_k,
        f'{ticker}_stoch_d': stoch_d
    }, index=close.index)

    # Drop any rows with NaN values from indicator calculations
    ticker_df.dropna(inplace=True)

    # Calculate stats (mean and std) for normalization
    mean_vals = ticker_df.mean()
    std_vals = ticker_df.std()

    for col in ticker_df.columns:
        stats[f'{col}_mean'] = mean_vals[col]
        stats[f'{col}_std'] = std_vals[col]

    # Normalize the DataFrame
    ticker_df = (ticker_df - mean_vals) / std_vals

    ticker_data_frames.append(ticker_df)

if not ticker_data_frames:
    raise ValueError("No valid data found for any tickers. Exiting.")

# Combine all data
df = pd.concat(ticker_data_frames, axis=1)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# -----------------------------
# 3) Create sequences and labels
# -----------------------------
all_sequences = []
all_labels = []

for ticker in tickers:
    needed_cols = [
        f'{ticker}_close', f'{ticker}_width', f'{ticker}_rsi',
        f'{ticker}_roc', f'{ticker}_volume', f'{ticker}_diff',
        f'{ticker}_pct_change', f'{ticker}_stoch_k', f'{ticker}_stoch_d'
    ]
    if any(col not in df.columns for col in needed_cols):
        continue
    
    features = df[needed_cols].values
    close = df[f'{ticker}_close'].values  # normalized close
    mean = stats[f'{ticker}_close_mean']
    std = stats[f'{ticker}_close_std']

    num_samples = len(features)
    for i in range(num_samples - sequence_length - future_gap + 1):
        seq = features[i:i+sequence_length]
        prev_close = close[i + sequence_length - 1]  # normalized
        next_close = close[i + sequence_length - 1 + future_gap]  # normalized
        # We'll store: [prev_close, next_close, mean, std]
        all_sequences.append(seq)
        all_labels.append([prev_close, next_close, mean, std])

all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

# Shuffle
indices = np.random.permutation(len(all_sequences))
all_sequences = all_sequences[indices]
all_labels = all_labels[indices]

# Split
train_size = int(0.9 * len(all_sequences))
val_size = int(0.05 * len(all_sequences))
test_size = len(all_sequences) - train_size - val_size

train_sequences = all_sequences[:train_size]
train_labels = all_labels[:train_size]

val_sequences = all_sequences[train_size:train_size + val_size]
val_labels = all_labels[train_size:train_size + val_size]

test_sequences = all_sequences[train_size + val_size:]
test_labels = all_labels[train_size + val_size:]

# -----------------------------
# 4) Dataset and DataLoaders
# -----------------------------
class StockDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

batch_size = 64
train_dataset = StockDataset(train_sequences, train_labels)
val_dataset = StockDataset(val_sequences, val_labels)
test_dataset = StockDataset(test_sequences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

# -----------------------------
# 5) Transformer Model
# -----------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim=9, d_model=128, nhead=8, 
                 dim_feedforward=1024, num_layers=4, dropout=0.2, 
                 sequence_length=24):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, sequence_length, d_model)
        )
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(d_model, 2)  # Predict mean and variance
    
    def forward(self, x):
        x = self.input_proj(x)             
        x = x + self.positional_embeddings 
        x = x.permute(1, 0, 2)             
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.permute(1, 2, 0)             
        x = self.pool(x).squeeze(-1)       
        x = self.norm(x)
        
        # Split into mean and variance components
        mean = x[:, 0]  # First output neuron
        var = torch.nn.functional.softplus(x[:, 1]) + 1e-6  # Ensure variance > 0
        return torch.stack([mean, var], dim=1)

# Move model to GPU
model = TransformerModel(input_dim=9).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------------
# 6) Custom Loss & Metrics
# -----------------------------

# -----------------------------
# 6) Modified Loss & Metrics
# -----------------------------
def gaussian_nll_loss(pred, labels):
    target = labels[:, 1]
    mean = pred[:, 0]
    var = pred[:, 1]
    return 0.5 * (torch.log(var) + (target - mean)**2 / var + math.log(2 * math.pi)).mean()

def custom_mae_loss(pred, labels):
    # labels format: [prev_close, next_close, mean, std]
    target = labels[:, 1]  # next_close (normalized)
    return torch.mean(torch.abs(pred.squeeze() - target))

def dir_acc(pred, labels):
    """Use only the mean prediction for direction accuracy"""
    mean_pred = pred[:, 0]  # Use mean component
    target = labels[:, 1]   # next_close (normalized)
    prev_close = labels[:, 0]  # prev_close (normalized)
    
    true_change = target - prev_close
    pred_change = mean_pred - prev_close
    correct = torch.sign(true_change) == torch.sign(pred_change)
    return correct.float().mean()

def magnitude_error(pred, labels):
    """Calculate error using mean prediction"""
    mean_pred = pred[:, 0].detach()
    prev_close = labels[:, 0] * labels[:, 3] + labels[:, 2]
    true_next = labels[:, 1] * labels[:, 3] + labels[:, 2]
    pred_next = mean_pred * labels[:, 3] + labels[:, 2]
    
    return torch.mean(torch.abs(true_next - pred_next))

def profit_metric(pred, labels):
    """
    If we 'buy' at prev_close and then 'sell' at next_close:
      - actual profit = true_next - prev_close
      - predicted profit = pred_next - prev_close
    We'll measure how close the predicted profit is to actual profit (MAE).
    """
    # Extract mean prediction only
    mean_pred = pred[:, 0]  # <-- This is the crucial fix
    
    prev_close = labels[:, 0] * labels[:, 3] + labels[:, 2]
    true_next = labels[:, 1] * labels[:, 3] + labels[:, 2]
    pred_next = mean_pred * labels[:, 3] + labels[:, 2]

    actual_profit = true_next - prev_close
    predicted_profit = pred_next - prev_close
    return torch.mean(torch.abs(actual_profit - predicted_profit))

# -----------------------------
# 7) Training Loop
# -----------------------------

def train_model(model, train_loader, val_loader, optimizer, epochs=10):
    best_loss = float('inf')
    train_losses, val_losses = [], []
    val_accs, val_mag_errors, val_profit_errors = [], [], []  # Added profit errors
    variance_stats = []  # To track variance metrics

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = gaussian_nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        total_mag_error, total_profit_error = 0, 0
        all_variances = []
        pred_bias = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Store variances and prediction bias
                all_variances.extend(outputs[:, 1].cpu().numpy())
                pred_bias += (outputs[:, 0] - labels[:, 1]).sum().item()

                # Compute metrics
                vloss = gaussian_nll_loss(outputs, labels).item()
                dacc = dir_acc(outputs, labels).item()
                magerr = magnitude_error(outputs, labels).item()
                perr = profit_metric(outputs, labels).item()

                val_loss += vloss * inputs.size(0)
                val_correct += dacc * inputs.size(0)
                total_mag_error += magerr * inputs.size(0)
                total_profit_error += perr * inputs.size(0)
                total += inputs.size(0)
        
        # Calculate validation metrics
        val_loss /= total
        val_acc = val_correct / total
        val_mag_err = total_mag_error / total
        val_profit_err = total_profit_error / total
        
        # Calculate variance stats
        variance_stats.append({
            'epoch': epoch+1,
            'mean_var': np.mean(all_variances),
            'median_var': np.median(all_variances),
            'min_var': np.min(all_variances),
            'max_var': np.max(all_variances)
        })
        
        # Calculate prediction bias
        avg_bias = pred_bias / total

        # Store metrics
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_mag_errors.append(val_mag_err)
        val_profit_errors.append(val_profit_err)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val DirAcc: {val_acc:.4f}")
        print(f"  Val MagErr ($): {val_mag_err:.4f} | Val ProfitErr ($): {val_profit_err:.4f}")
        print(f"  Variance Stats: Mean={variance_stats[-1]['mean_var']:.4f}, "
              f"Median={variance_stats[-1]['median_var']:.4f}")
        print(f"  Prediction Bias: {avg_bias:.4f} "
              f"(Positive = overestimating, Negative = underestimating)")

    return train_losses, val_losses, val_accs, val_mag_errors, val_profit_errors

# -----------------------------
# 8) Run Training
# -----------------------------
train_losses, val_losses, val_accs, val_mag_errors, val_profit_errors = train_model(
    model, train_loader, val_loader, optimizer, epochs=100
)

# -----------------------------
# 9) Evaluation on Test Set
# -----------------------------
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)

def evaluate_model(model, test_loader):
    model.eval()
    test_loss, test_correct, total = 0, 0, 0
    predictions = []
    true_values = []
    variances = []
    means = []
    stds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Store predictions and normalization parameters
            mean_pred = outputs[:, 0].cpu().numpy()
            var_pred = outputs[:, 1].cpu().numpy()
            true_next = labels[:, 1].cpu().numpy()
            batch_means = labels[:, 2].cpu().numpy()
            batch_stds = labels[:, 3].cpu().numpy()
            
            predictions.extend(mean_pred)
            true_values.extend(true_next)
            variances.extend(var_pred)
            means.extend(batch_means)
            stds.extend(batch_stds)
            
            # Compute metrics
            test_loss += gaussian_nll_loss(outputs, labels).item() * inputs.size(0)
            test_correct += dir_acc(outputs, labels).item() * inputs.size(0)
            total += inputs.size(0)
    
    test_loss /= total
    test_acc = test_correct / total
    
    # Correct denormalization using per-sample statistics
    denorm_pred = np.array(predictions) * np.array(stds) + np.array(means)
    denorm_true = np.array(true_values) * np.array(stds) + np.array(means)
    denorm_var = np.array(variances) * (np.array(stds)**2)
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'predictions': denorm_pred,
        'true_values': denorm_true,
        'variances': denorm_var,
        'timestamps': df.index[sequence_length + future_gap - 1:len(df) - future_gap + 1]
    }

# After evaluation
results = evaluate_model(model, test_loader)

# -----------------------------
# New: Continuous Prediction Visualization
# -----------------------------
def plot_continuous_predictions(ticker='AAPL', lookback=500):
    # Extract data for specific ticker
    ticker_cols = [c for c in df.columns if c.startswith(f'{ticker}_')]
    ticker_data = df[ticker_cols].dropna()
    
    # Create continuous sequences
    cont_sequences = []
    cont_labels = []
    
    features = ticker_data.values
    close = ticker_data[f'{ticker}_close'].values
    mean = stats[f'{ticker}_close_mean']
    std = stats[f'{ticker}_close_std']
    
    for i in range(len(features) - sequence_length - future_gap + 1):
        seq = features[i:i+sequence_length]
        prev_close = close[i + sequence_length - 1]
        next_close = close[i + sequence_length - 1 + future_gap]
        cont_sequences.append(seq)
        cont_labels.append([prev_close, next_close, mean, std])
    
    # Create dataset and predict
    cont_dataset = StockDataset(np.array(cont_sequences), np.array(cont_labels))
    cont_loader = DataLoader(cont_dataset, batch_size=64, shuffle=False)
    
    # Get predictions
    model.eval()
    all_preds, all_trues, all_vars = [], [], []
    timestamps = []
    with torch.no_grad():
        for inputs, labels in cont_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Denormalize using per-sample stats
            preds = outputs[:, 0].cpu().numpy() * labels[:, 3].numpy() + labels[:, 2].numpy()
            trues = labels[:, 1].numpy() * labels[:, 3].numpy() + labels[:, 2].numpy()
            vars = outputs[:, 1].cpu().numpy() * (labels[:, 3].numpy()**2)
            
            all_preds.extend(preds)
            all_trues.extend(trues)
            all_vars.extend(vars)
    
    # Get corresponding timestamps
    timestamps = ticker_data.index[sequence_length + future_gap - 1:len(ticker_data) - future_gap + 1]
    
    # Plot last N points
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps[-lookback:], all_trues[-lookback:], label='Actual Price', linewidth=1)
    plt.plot(timestamps[-lookback:], all_preds[-lookback:], label='Predicted Price', alpha=0.8)
    plt.fill_between(timestamps[-lookback:], 
                     np.array(all_preds[-lookback:]) - 2*np.sqrt(np.array(all_vars[-lookback:])),
                     np.array(all_preds[-lookback:]) + 2*np.sqrt(np.array(all_vars[-lookback:])),
                     alpha=0.2, label='95% Confidence')
    plt.title(f'{ticker} Price Prediction - Last {lookback} Periods')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 9) Evaluation and Visualization
# -----------------------------
# Plot test results
plt.figure(figsize=(10, 6))
plt.plot(results['predictions'][:100], label='Predicted')
plt.plot(results['true_values'][:100], label='True')
plt.fill_between(range(100),
                 results['predictions'][:100] - 2*np.sqrt(results['variances'][:100]),
                 results['predictions'][:100] + 2*np.sqrt(results['variances'][:100]),
                 alpha=0.2, label='95% Confidence')
plt.legend()
plt.title('Random Sample Predictions with Uncertainty Bands')
plt.show()

# Plot continuous timeline for specific ticker
plot_continuous_predictions(ticker='AAPL', lookback=500)

# Training metrics plots remain the same
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Train vs. Val Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_accs, label='Val Directional Acc')
plt.title('Directional Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(val_mag_errors, label='Price Error')
plt.plot(val_profit_errors, label='Profit Error')
plt.title('Error Metrics ($)')
plt.legend()

plt.tight_layout()
plt.show()