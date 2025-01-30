import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import List, Union, Optional
import pytz
import hashlib
import pickle
from typing import List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class StockData:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        cache_dir: str = "stock_data"
    ):
        """
        Initialize StockData manager with Alpaca credentials and cache directory.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: Alpaca API base URL
            cache_dir: Directory for storing cached data
        """
        self.client = StockHistoricalDataClient(api_key, api_secret)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.info(f"Initialized StockData with cache directory: {self.cache_dir}")

    def _get_cache_path(self, symbol: str, timeframe: TimeFrame) -> str:
        """Generate cache file path for given symbol and timeframe."""
        tf_str = f"{timeframe.amount}{timeframe.unit.name}"
        symbol_dir = os.path.join(self.cache_dir, symbol.upper())
        os.makedirs(symbol_dir, exist_ok=True)
        return os.path.join(symbol_dir, f"{tf_str}.csv")

    def _load_cached_data(self, symbol: str, timeframe: TimeFrame) -> Optional[pd.DataFrame]:
        """Load cached data for symbol/timeframe if exists."""
        cache_path = self._get_cache_path(symbol, timeframe)
        if not os.path.exists(cache_path):
            return None
        try:
            df = pd.read_csv(cache_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            if df.index.tz is None:
                # If the datetime index is naive, localize it to UTC
                df.index = df.index.tz_localize(pytz.UTC)
                logging.debug(f"Timestamps for {symbol} localized to UTC.")
            else:
                # If the datetime index is already timezone-aware, convert it to UTC
                df.index = df.index.tz_convert(pytz.UTC)
                logging.debug(f"Timestamps for {symbol} converted to UTC.")
            
            return df.sort_index()
        except Exception as e:
            logging.error(f"Error loading cache for {symbol}: {e}")
            return None

    def _save_to_cache(self, symbol: str, timeframe: TimeFrame, df: pd.DataFrame):
        """Save DataFrame to cache, merging with existing data if needed."""
        cache_path = self._get_cache_path(symbol, timeframe)
        
        # Explicitly check if cached data is None
        existing_df = self._load_cached_data(symbol, timeframe)
        if existing_df is None:
            existing_df = pd.DataFrame()
        
        # Alternatively, using a one-liner with a conditional expression
        # existing_df = self._load_cached_data(symbol, timeframe) if self._load_cached_data(symbol, timeframe) is not None else pd.DataFrame()
        
        combined_df = pd.concat([existing_df, df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        
        # Ensure all timestamps are in UTC before saving
        if combined_df.index.tz is not None:
            combined_df.index = combined_df.index.tz_convert(pytz.UTC)
        else:
            combined_df.index = combined_df.index.tz_localize(pytz.UTC)
        
        combined_df.reset_index().to_csv(cache_path, index=False)
        logging.info(f"Updated cache for {symbol} ({timeframe}) at {cache_path}")

    def get_data(
        self,
        symbols: Union[str, List[str]],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        adjustment: str = 'all'
    ) -> pd.DataFrame:
        """
        Get historical stock data with smart caching.
        
        Args:
            symbols: Ticker symbol(s) to fetch
            timeframe: Bar timeframe (e.g., 5Min, 1Hour)
            start: Start datetime (UTC, inclusive)
            end: End datetime (UTC, exclusive)
            adjustment: Data adjustment type (raw, split, dividend, all)
        
        Returns:
            Combined DataFrame with historical data
        """
        symbols = [symbols] if isinstance(symbols, str) else symbols
        start = start.astimezone(pytz.UTC)
        end = min(end.astimezone(pytz.UTC), datetime.now(pytz.UTC))
        
        all_data = []
        for symbol in symbols:
            # Load existing cache and determine missing ranges
            cached_df = self._load_cached_data(symbol, timeframe)
            missing_ranges = self._get_missing_ranges(cached_df, start, end)
            
            # Fetch missing data
            new_data = []
            for range_start, range_end in missing_ranges:
                logging.info(f"Fetching {symbol} {timeframe} data from {range_start} to {range_end}")
                df = self._fetch_alpaca_data(symbol, timeframe, range_start, range_end, adjustment)
                if not df.empty:
                    new_data.append(df)
            
            # Update cache with new data
            if new_data:
                combined_new = pd.concat(new_data)
                self._save_to_cache(symbol, timeframe, combined_new)
                cached_df = self._load_cached_data(symbol, timeframe)
            
            # Collect requested data
            if cached_df is not None:
                filtered = cached_df.loc[start:end]
                all_data.append(filtered)
        
        return pd.concat(all_data).sort_index() if all_data else pd.DataFrame()

    def _get_missing_ranges(
        self,
        cached_df: Optional[pd.DataFrame],
        req_start: datetime,
        req_end: datetime
    ) -> List[tuple]:
        """Identify time ranges not covered by cached data."""
        if cached_df is None or cached_df.empty:
            return [(req_start, req_end)]
        
        cache_start = cached_df.index.min()
        cache_end = cached_df.index.max()
        missing = []

        # Missing data before cache
        if req_start < cache_start:
            missing.append((req_start, min(cache_start, req_end)))
        
        # Missing data after cache
        if req_end > cache_end:
            missing.append((max(cache_end, req_start), req_end))
        
        return missing

    def _fetch_alpaca_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        adjustment: str
    ) -> pd.DataFrame:
        """Fetch data from Alpaca API with pagination handling."""
        all_bars = []
        page_token = None
        
        while True:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    adjustment=adjustment,
                    page_token=page_token,
                    limit=10000
                )
                response = self.client.get_stock_bars(request)
                bars = response.data.get(symbol, [])
                
                # Process bars
                for bar in bars:
                    all_bars.append({
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'symbol': symbol
                    })
                
                # Check for pagination
                page_token = getattr(response, 'next_page_token', None)
                if not page_token:
                    break
            except Exception as e:
                logging.error(f"Error fetching {symbol} data: {e}")
                break
        
        if not all_bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_bars)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(pytz.UTC)
        df.set_index('timestamp', inplace=True)
        return df.sort_index().drop_duplicates()

    def update_data(
        self,
        symbols: Union[str, List[str]],
        timeframe: TimeFrame,
        lookback: timedelta = timedelta(days=7),
        adjustment: str = 'all'
    ) -> pd.DataFrame:
        """
        Update cached data with latest information.
        
        Args:
            symbols: Ticker symbol(s) to update
            timeframe: Bar timeframe to update
            lookback: Period to fetch if no cached data exists
            adjustment: Data adjustment type
        
        Returns:
            Updated DataFrame with latest data
        """
        symbols = [symbols] if isinstance(symbols, str) else symbols
        end = datetime.now(pytz.UTC)
        all_data = []
        
        for symbol in symbols:
            cached_df = self._load_cached_data(symbol, timeframe)
            start = cached_df.index.max() + timedelta(seconds=1) if cached_df is not None else end - lookback
            
            if start >= end:
                logging.info(f"{symbol} data is already up-to-date")
                continue
                
            new_data = self._fetch_alpaca_data(symbol, timeframe, start, end, adjustment)
            if not new_data.empty:
                self._save_to_cache(symbol, timeframe, new_data)
                all_data.append(new_data)
        
        return pd.concat(all_data).sort_index() if all_data else pd.DataFrame()

class StockDataProcessor:
    def __init__(
        self,
        stock_data: 'StockData',  # Replace with actual StockData import
        processed_dir: str = "processed_data",
        indicators: Optional[List[str]] = None
    ):
        """
        Initialize StockDataProcessor with a StockData instance, processed data directory, and indicators.
        
        Args:
            stock_data: Instance of StockData for fetching stock data
            processed_dir: Directory for storing processed data
            indicators: List of indicators to compute
        """
        self.stock_data = stock_data
        self.processed_dir = processed_dir
        self.indicators = indicators if indicators is not None else [
            'Bollinger_Bands',
            'RSI',
            'ROC',
            'Stochastic_Oscillator'
        ]
        os.makedirs(self.processed_dir, exist_ok=True)
        logging.info(f"Initialized StockDataProcessor with directory: {self.processed_dir} and indicators: {self.indicators}")

    def _generate_hash(self, df: pd.DataFrame) -> str:
        """
        Generate a SHA256 hash based on the first row of the DataFrame.
        
        Args:
            df: DataFrame to hash
        
        Returns:
            Hexadecimal hash string
        """
        first_row = df.iloc[0].to_json()
        hash_obj = hashlib.sha256(first_row.encode('utf-8'))
        return hash_obj.hexdigest()

    def _get_processed_path(self, symbol: str, hash_str: str) -> str:
        """Generate the file path for processed data based on symbol and hash."""
        symbol_dir = os.path.join(self.processed_dir, symbol.upper())
        os.makedirs(symbol_dir, exist_ok=True)
        return os.path.join(symbol_dir, f"{hash_str}.csv")

    def _load_processed_data(self, symbol: str, hash_str: str) -> Optional[pd.DataFrame]:
        """Load processed data if it exists."""
        processed_path = self._get_processed_path(symbol, hash_str)
        if os.path.exists(processed_path):
            try:
                df = pd.read_csv(processed_path, parse_dates=['timestamp'])
                df.set_index('timestamp', inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize(pytz.UTC)
                else:
                    df.index = df.index.tz_convert(pytz.UTC)
                logging.info(f"Loaded processed data for {symbol} from {processed_path}")
                return df.sort_index()
            except Exception as e:
                logging.error(f"Error loading processed data for {symbol}: {e}")
                return None
        return None

    def _save_processed_data(self, df: pd.DataFrame, symbol: str, hash_str: str):
        """Save the processed DataFrame to a CSV file."""
        processed_path = self._get_processed_path(symbol, hash_str)
        df.reset_index().to_csv(processed_path, index=False)
        logging.info(f"Saved processed data for {symbol} to {processed_path}")

    def process_data(
        self,
        symbols: Union[str, List[str]],
        timeframe: 'TimeFrame',  # Replace with actual TimeFrame import
        start: datetime,
        end: datetime,
        adjustment: str = 'all',
        moving_average_window: int = 20
    ) -> pd.DataFrame:
        """
        Process stock data by computing statistics and caching the results.
        
        Args:
            symbols: Ticker symbol(s) to process
            timeframe: Bar timeframe (e.g., 5Min, 1Hour)
            start: Start datetime (UTC, inclusive)
            end: End datetime (UTC, exclusive)
            adjustment: Data adjustment type (raw, split, dividend, all)
            moving_average_window: Window size for moving average
            
        Returns:
            Combined DataFrame with additional statistical columns
        """
        symbols = [symbols] if isinstance(symbols, str) else symbols
        all_processed_data = []

        for symbol in symbols:
            logging.info(f"Processing data for {symbol}...")
            # Fetch raw data for the symbol
            raw_data = self.stock_data.get_data(
                symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                adjustment=adjustment
            )

            if 'symbol' in raw_data.columns:
                # Remove symbol column
                raw_data.drop(columns=['symbol'], inplace=True)
            # Remove any rows with NaN values in raw data
            raw_data.dropna(inplace=True)

            if raw_data.empty:
                logging.warning(f"No raw data available for {symbol} to process.")
                continue

            # Generate hash based on the first row of the symbol's data
            hash_str = self._generate_hash(raw_data)
            
            # Check if processed data already exists
            processed_df = self._load_processed_data(symbol, hash_str)
            if processed_df is not None:
                all_processed_data.append(processed_df)
                continue

            # Perform processing (compute indicators)
            logging.info(f"Computing technical indicators for {symbol}...")
            processed_df = raw_data.copy()
            
            # Calculate Moving Average
            processed_df['MA_20'] = self._calculate_moving_average(processed_df['close'], window=moving_average_window)
            
            # Calculate Bollinger Bands
            bollinger = self._calculate_bollinger_bands(processed_df['close'], window=20, num_of_std=2)
            processed_df = processed_df.join(bollinger)
            
            # Calculate RSI
            processed_df['RSI_14'] = self._calculate_rsi(processed_df['close'], window=14)
            
            # Calculate ROC
            processed_df['ROC_14'] = self._calculate_roc(processed_df['close'], periods=14)
            
            # Calculate Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic_oscillator(
                processed_df['high'], processed_df['low'], processed_df['close'],
                k_period=14, d_period=3
            )
            processed_df['Stoch_%K'] = stoch_k
            processed_df['Stoch_%D'] = stoch_d
            
            # Drop rows with any NaN values resulting from indicator calculations
            before_drop = len(processed_df)
            processed_df.dropna(inplace=True)
            after_drop = len(processed_df)
            dropped_rows = before_drop - after_drop
            if dropped_rows > 0:
                logging.info(f"Dropped {dropped_rows} rows with incomplete indicators for {symbol}.")
            
            if processed_df.empty:
                logging.warning(f"All data was dropped for {symbol} due to incomplete indicators.")
                continue
            
            # Compute Log Returns
            logging.info(f"Computing log returns for {symbol}...")
            processed_df['log_return'] = self._calculate_log_return(processed_df['close'])
            
            # After computing log returns, drop the first row which will have NaN log_return
            initial_length = len(processed_df)
            processed_df.dropna(subset=['log_return'], inplace=True)
            final_length = len(processed_df)
            if final_length < initial_length:
                logging.info(f"Dropped {initial_length - final_length} rows due to NaN log returns for {symbol}.")

            if processed_df.empty:
                logging.warning(f"All data was dropped for {symbol} after computing log returns.")
                continue
            
            # Save processed data
            self._save_processed_data(processed_df, symbol, hash_str)
            all_processed_data.append(processed_df)
        
        return pd.concat(all_processed_data).sort_index() if all_processed_data else pd.DataFrame()

    # --- Internal Indicator Methods ---
    def _calculate_moving_average(self, data: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Moving Average.
        
        Args:
            data: Pandas Series of closing prices.
            window: The number of periods for the moving average.
        
        Returns:
            Pandas Series representing the moving average.
        """
        return data.rolling(window=window).mean()

    def _calculate_bollinger_bands(self, data: pd.Series, window: int = 20, num_of_std: int = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Pandas Series of closing prices.
            window: The number of periods for the moving average.
            num_of_std: The number of standard deviations for the bands.
        
        Returns:
            A DataFrame with 'Bollinger_Upper', 'Bollinger_Lower', and 'Bollinger_Width'.
        """
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_of_std)
        lower_band = rolling_mean - (rolling_std * num_of_std)
        width = upper_band - lower_band
        return pd.DataFrame({
            'Bollinger_Upper': upper_band,
            'Bollinger_Lower': lower_band,
            'Bollinger_Width': width
        })

    def _calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Pandas Series of closing prices.
            window: The number of periods to use for RSI calculation.
        
        Returns:
            A Pandas Series representing RSI.
        """
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_roc(self, data: pd.Series, periods: int = 14) -> pd.Series:
        """
        Calculate Rate of Change (ROC).
        
        Args:
            data: Pandas Series of closing prices.
            periods: The number of periods to use for ROC calculation.
        
        Returns:
            A Pandas Series representing ROC.
        """
        roc = ((data - data.shift(periods)) / data.shift(periods)) * 100
        return roc

    def _calculate_stochastic_oscillator(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> (pd.Series, pd.Series):
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        Args:
            high: Pandas Series of high prices.
            low: Pandas Series of low prices.
            close: Pandas Series of closing prices.
            k_period: The number of periods for %K.
            d_period: The number of periods for %D.
        
        Returns:
            Two Pandas Series representing %K and %D.
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_line = k_line.rolling(window=d_period).mean()
        return k_line, d_line

    def _calculate_log_return(self, close_prices: pd.Series) -> pd.Series:
        """
        Calculate Log Returns based on closing prices.
        
        Args:
            close_prices: Pandas Series of closing prices.
        
        Returns:
            Pandas Series representing log returns.
        """
        return np.log(close_prices / close_prices.shift(1))

class StockDatasetGenerator:
    def __init__(
        self,
        stock_data_processor,  # Replace with StockDataProcessor if defined
        input_sequence_length: int = 24,
        prediction_horizon: int = 12,
        dataset_save_path: str = 'stock_dataset.pkl',
        scalers_save_path: str = 'scalers.pkl',
        train_split: float = 0.8,
        validation_split: float = 0.1,
        test_split: float = 0.1
    ):
        """
        Initialize the StockDatasetGenerator.

        Args:
            stock_data_processor: Instance of StockDataProcessor for data processing.
            input_sequence_length: Number of past time steps to use as input.
            prediction_horizon: Number of time steps ahead to predict.
            dataset_save_path: File path to save the generated dataset.
            scalers_save_path: File path to save the per-stock scalers.
            train_split: Proportion of data to be used for training.
            validation_split: Proportion of data to be used for validation.
            test_split: Proportion of data to be used for testing.
        """
        self.processor = stock_data_processor
        self.input_sequence_length = input_sequence_length
        self.prediction_horizon = prediction_horizon
        self.dataset_save_path = dataset_save_path
        self.scalers_save_path = scalers_save_path
        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split

        # Initialize separate datasets
        self.train_dataset: List[Tuple[pd.DataFrame, pd.Series, str]] = []
        self.validation_dataset: List[Tuple[pd.DataFrame, pd.Series, str]] = []
        self.test_dataset: List[Tuple[pd.DataFrame, pd.Series, str]] = []

        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, List[str]] = {}

        logging.info("Initialized StockDatasetGenerator with per-stock normalization and dataset splits.")

    def generate_dataset(
        self,
        symbols: Union[str, List[str]],
        timeframe,
        start: datetime,
        end: datetime,
        adjustment: str = 'all',
        moving_average_window: int = 20
    ):
        symbols = [symbols] if isinstance(symbols, str) else symbols
        logging.info(f"Starting dataset generation for symbols: {symbols}")

        for symbol in symbols:
            logging.info(f"Processing symbol: {symbol}")
            processed_df = self.processor.process_data(
                symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                adjustment=adjustment,
                moving_average_window=moving_average_window
            )

            if processed_df.empty:
                logging.warning(f"No processed data for {symbol}. Skipping.")
                continue

            # Compute Relative Features
            relative_features = {
                'rel_open': 'open',
                'rel_high': 'high',
                'rel_low': 'low',
                'rel_close': 'close',
                'rel_boll_upper': 'Bollinger_Upper',
                'rel_boll_lower': 'Bollinger_Lower',
                'rel_boll_width': 'Bollinger_Width',
            }

            for new_col, orig_col in relative_features.items():
                if orig_col in processed_df.columns and 'MA_20' in processed_df.columns:
                    # Avoid division by zero by adding a small epsilon if needed
                    epsilon = 1e-8
                    processed_df[new_col] = processed_df[orig_col] / (processed_df['MA_20'] + epsilon)
                else:
                    logging.warning(f"Column {orig_col} or MA20 not found in {symbol} data. Skipping relative feature {new_col}.")

            # Add Labels (Optional if not handled elsewhere)
            # If you need to add a label column indicating the feature type, you can do so here.
            # For example, creating a multi-index or adding suffixes as done above suffices for labeling.

            # Add relative features to the DataFrame
            # (Already added above)

            # Add additional engineered features
            processed_df = self._add_total_log_returns(processed_df)
            processed_df = self._add_positional_encodings(processed_df)

            
            # Remove Absolute Features and MA20
            absolute_columns = ['open', 'high', 'low', 'close', 'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Width', 'MA_20']
            missing_cols = [col for col in absolute_columns if col not in processed_df.columns]
            if missing_cols:
                logging.warning(f"Some columns to drop are missing in {symbol} data: {missing_cols}")
            processed_df = processed_df.drop(columns=[col for col in absolute_columns if col in processed_df.columns])

            # Remove initial NaNs from total_log_return
            valid_start = self.prediction_horizon
            processed_df = processed_df.iloc[valid_start:].reset_index(drop=True)
            

            # Split into train/val/test sets
            total_length = len(processed_df)
            train_end = int(total_length * self.train_split)
            val_end = train_end + int(total_length * self.validation_split)

            train_data = processed_df.iloc[:train_end]
            val_data = processed_df.iloc[train_end:val_end]
            test_data = processed_df.iloc[val_end:]

            # Fit scaler only on training data
            features_train = train_data.drop(columns=['timestamp', 'symbol'], errors='ignore')
            feature_scaler = StandardScaler()
            feature_scaler.fit(features_train)

            # Adjust cyclic features scaling
            cyclic_features = ['month_cos', 'month_sin', 'day_of_week_cos', 'day_of_week_sin', 'hour_cos', 'hour_sin']
            for cf in cyclic_features:
                if cf in features_train.columns:
                    idx = list(features_train.columns).index(cf)
                    feature_scaler.mean_[idx] = 0.0
                    feature_scaler.scale_[idx] = 1.0
            self.scalers[symbol.upper()] = feature_scaler

            # Store feature names
            self.feature_names[symbol.upper()] = features_train.columns.tolist()

            # Process each split
            for split_name, split_data in [('train', train_data),
                                         ('validation', val_data),
                                         ('test', test_data)]:
                if split_data.empty:
                    continue

                # Scale features using training scaler
                features_split = split_data.drop(columns=['timestamp', 'symbol'], errors='ignore')
                scaled_features = feature_scaler.transform(features_split)
                scaled_df = pd.DataFrame(scaled_features, columns=features_split.columns)

                # Generate sequences
                split_samples = []
                num_samples = len(scaled_df) - self.input_sequence_length - self.prediction_horizon + 1
                for i in range(num_samples):
                    input_start = i
                    input_end = i + self.input_sequence_length
                    output_index = i + self.input_sequence_length + self.prediction_horizon - 1

                    input_sequence = scaled_df.iloc[input_start:input_end]
                    output_row = scaled_df.iloc[output_index]

                    split_samples.append((
                        input_sequence.reset_index(drop=True),
                        output_row,
                        symbol.upper()
                    ))

                # Add to corresponding dataset
                if split_name == 'train':
                    self.train_dataset.extend(split_samples)
                elif split_name == 'validation':
                    self.validation_dataset.extend(split_samples)
                else:
                    self.test_dataset.extend(split_samples)

                logging.info(f"Added {len(split_samples)} {split_name} samples for {symbol}")

        logging.info(f"Total samples - Train: {len(self.train_dataset)}, Val: {len(self.validation_dataset)}, Test: {len(self.test_dataset)}")

    def _add_total_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add total_log_return feature calculated over prediction_horizon period."""
        df = df.copy()
        
        # Calculate total_log_return over the prediction_horizon
        df['total_log_return'] = np.log(df['close'] / df['close'].shift(self.prediction_horizon))
        
        return df

    def _add_positional_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cosine and sine positional encodings for month, day of week, and hour of day.

        Args:
            df: DataFrame containing at least a 'timestamp' column.

        Returns:
            DataFrame with added positional encoding features.
        """
        df = df.copy()
        # make sure df.index is timestmap
        if df.index is None:
            raise ValueError("Index of the DataFrame is not set to timestamp")
        
        timestamp = df.index
        # Ensure 'timestamp' is a datetime object
        timestamp = pd.to_datetime(timestamp)

        # Extract components
        df['month'] = timestamp.month
        df['day_of_week'] = timestamp.dayofweek
        df['hour'] = timestamp.hour

        # Define maximum values for normalization
        max_month = 12
        max_day = 7
        max_hour = 24

        # Compute positional encodings
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / max_month)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / max_month)

        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / max_day)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / max_day)

        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / max_hour)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / max_hour)

        # Drop the original categorical columns
        df.drop(columns=['month', 'day_of_week', 'hour'], inplace=True)

        logging.info("Added positional encodings for month, day_of_week, and hour.")

        return df

    def save_dataset(self):
        """Save train, validation, test datasets, scalers, and feature names with proper type checking."""
        # Ensure directories exist
        dataset_dir = os.path.dirname(self.dataset_save_path)
        scaler_dir = os.path.dirname(self.scalers_save_path)
        if dataset_dir:
            os.makedirs(dataset_dir, exist_ok=True)
        if scaler_dir:
            os.makedirs(scaler_dir, exist_ok=True)

        # Prepare data to save
        dataset = {
            'train': self.train_dataset,
            'validation': self.validation_dataset,
            'test': self.test_dataset,
            'feature_names': self.feature_names  # Include feature names
        }

        with open(self.dataset_save_path, 'wb') as f:
            pickle.dump(dataset, f)
        logging.info(f"Datasets saved to {self.dataset_save_path}")

        # Save scalers separately
        with open(self.scalers_save_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logging.info(f"Scalers saved to {self.scalers_save_path}")

    def load_dataset(self):
        """Load train, validation, test datasets, scalers, and feature names with error handling."""
        if not os.path.exists(self.dataset_save_path):
            raise FileNotFoundError(f"Dataset file {self.dataset_save_path} not found")

        if not os.path.exists(self.scalers_save_path):
            raise FileNotFoundError(f"Scalers file {self.scalers_save_path} not found")

        with open(self.dataset_save_path, 'rb') as f:
            dataset = pickle.load(f)
            self.train_dataset = dataset.get('train', [])
            self.validation_dataset = dataset.get('validation', [])
            self.test_dataset = dataset.get('test', [])
            self.feature_names = dataset.get('feature_names', {})  # Load feature names

        with open(self.scalers_save_path, 'rb') as f:
            self.scalers = pickle.load(f)

        logging.info("Successfully loaded train, validation, test datasets, scalers, and feature names")

    def get_train_dataset(self) -> List[Tuple[pd.DataFrame, pd.Series, str]]:
        """Get the training dataset."""
        return self.train_dataset

    def get_validation_dataset(self) -> List[Tuple[pd.DataFrame, pd.Series, str]]:
        """Get the validation dataset."""
        return self.validation_dataset

    def get_test_dataset(self) -> List[Tuple[pd.DataFrame, pd.Series, str]]:
        """Get the testing dataset."""
        return self.test_dataset

    def get_scaler(self, symbol: str) -> StandardScaler:
        """Get the scaler for a specific symbol."""
        return self.scalers.get(symbol.upper())

    def get_feature_names(self, symbol: str) -> List[str]:
        """Get the feature names for a specific symbol."""
        return self.feature_names.get(symbol.upper(), [])

def visualize_samples(dataset_generator, num_samples=3, ticker='AAPL'):
    # Get denormalization parameters
    scaler = dataset_generator.get_scaler(ticker)
    
    # Filter samples for specific ticker
    ticker_samples = [s for s in dataset_generator.train_dataset if s[2] == ticker]
    
    if not ticker_samples:
        print(f"No samples found for ticker {ticker}")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # Plot full timeline of predictions
    all_predictions = []
    input_features = []

    for idx, (input_seq, output, _) in enumerate(ticker_samples[:num_samples]):
        # Denormalize data
        denorm_input = scaler.inverse_transform(input_seq)
        denorm_output = scaler.inverse_transform(output.to_frame().T)
        
        # Convert to DataFrame for easier handling
        input_df = pd.DataFrame(denorm_input, columns=input_seq.columns)
        output_df = pd.DataFrame(denorm_output, columns=output.index)
        
        # Store values for plotting
        all_predictions.append(output_df['total_log_return'].values[0])
        input_features.append(input_df['close'].values)
    
    # Plot input sequences
    for i, closes in enumerate(input_features):
        ax1.plot(range(len(closes)), closes, label=f'Sample {i+1} Input', alpha=0.7)
        ax1.axvline(x=len(closes)-1, color='gray', linestyle='--')  # Mark prediction point
    
    ax1.set_title('Input Sequences (Close Price)')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot predictions without timeline
    ax2.plot(range(len(all_predictions)), all_predictions, 'o-', label='Predicted Total Log Returns')
    ax2.set_title('Predicted Total Log Returns')
    ax2.set_ylabel('Log Return')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_feature_progression(dataset_generator, feature_name='total_log_return', ticker='AAPL'):
    """Plot the progression of a specific feature across all samples"""
    scaler = dataset_generator.get_scaler(ticker)
    ticker_samples = [s for s in dataset_generator.dataset if s[2] == ticker]
    
    if not ticker_samples:
        print(f"No samples found for ticker {ticker}")
        return

    # Collect all feature values
    input_values = []
    output_values = []
    indices = []

    for idx, (input_seq, output, _) in enumerate(ticker_samples):
        # Denormalize
        denorm_input = scaler.inverse_transform(input_seq)
        denorm_output = scaler.inverse_transform(output.to_frame().T)
        
        # Get feature values
        input_feature = denorm_input[:, input_seq.columns.get_loc(feature_name)]
        output_feature = denorm_output[0, output.index.get_loc(feature_name)]
        
        input_values.append(input_feature)
        output_values.append(output_feature)
        indices.append(idx)

    # Create plot
    plt.figure(figsize=(14, 6))
    
    # Plot input features as shaded areas
    for i, vals in enumerate(input_values):
        plt.plot(range(len(vals)), vals, alpha=0.2, color='blue')
    
    # Plot output predictions
    plt.plot(indices, output_values, 'ro-', label='Predictions')
    
    plt.title(f'{feature_name} Progression')
    plt.ylabel(feature_name)
    plt.xlabel('Sample Index')
    plt.grid(True)
    plt.legend()
    plt.show()   
    
# Example Usage
if __name__ == "__main__":
    API_KEY = "AKUXM8FH9NXFHBMBJ42M"
    API_SECRET = "OfnzmCT4lafIAnQySAtGLGg7buWN0y2CIufYGVwh"
    BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    stock_data = StockData(API_KEY, API_SECRET, BASE_URL)

    # Initialize StockDataProcessor
    processor = StockDataProcessor(stock_data)

    # Initialize StockDatasetGenerator
    dataset_generator = StockDatasetGenerator(
        stock_data_processor=processor,
        input_sequence_length=24,
        prediction_horizon=12,
        dataset_save_path='data/stock_dataset.pkl',  # Specify directory
        scalers_save_path='data/scalers.pkl',
        train_split=0.8,
        validation_split=0.1,
        test_split=0.1
    )

    # Define parameters
    timeframe = TimeFrame(5, TimeFrameUnit.Minute)
    start = datetime.utcnow() - timedelta(days=365)
    end = datetime.utcnow()
    symbols = ["AAPL"]

    # Generate the dataset
    dataset_generator.generate_dataset(
        symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment='all',
        moving_average_window=20
    )

    # Save the dataset and scalers
    dataset_generator.save_dataset()

    # Optionally, load the dataset and scalers later
    dataset_generator.load_dataset()
    # scalers = dataset_generator.get_scaler()

    # Display some samples
    if dataset_generator.train_dataset:
        sample_input, sample_output, ticker = dataset_generator.train_dataset[0]
        print("Sample Input Sequence:")
        print(sample_input)
        print("\nSample Output Row:")
        print(sample_output)
    else:
        print("No samples were generated.")

    # visualize_samples(dataset_generator, num_samples=5, ticker='AAPL')
    # plot_feature_progression(dataset_generator, feature_name='total_log_return', ticker='AAPL')

    # # Extract all 'total_log_return' values from the dataset
    # total_log_returns_scaled = np.array([sample[1]['total_log_return'] for sample in dataset]).reshape(-1, 1)

    # # Retrieve the scaler for 'total_log_return'
    # # total_log_return_scaler = scalers.get('total_log_return')

    # if total_log_return_scaler is None:
    #     raise ValueError("'total_log_return' scaler not found in scalers.pkl")

    # # Inverse transform to get original 'total_log_return' values
    # total_log_returns = total_log_return_scaler.inverse_transform(total_log_returns_scaled).flatten()

    # # Create a DataFrame for easier handling
    # log_return_df = pd.DataFrame(total_log_returns, columns=['Total Log Return'])

    # # Plot the distribution
    # plt.figure(figsize=(12, 6))
    # sns.histplot(log_return_df['Total Log Return'], bins=100, kde=True, color='teal')
    # plt.title('Distribution of Total Log Return')
    # plt.xlabel('Total Log Return')
    # plt.ylabel('Frequency')
    # plt.show()

    # # Perform a normality test
    # from scipy.stats import normaltest

    # stat, p_value = normaltest(total_log_returns)
    # print(f'Normality Test Stat: {stat:.4f}, P-value: {p_value:.4f}')
    # if p_value > 0.05:
    #     print("Total log returns are likely normally distributed.")
    # else:
    #     print("Total log returns are likely not normally distributed.")