# -*- coding: utf-8 -*-
# improved_stock_cnn2d_pytorch.py
"""
Improvements over the original version:
1. Enhanced feature engineering with technical indicators
2. Better CNN architecture with attention mechanism
3. Improved data preprocessing and normalization
4. Better handling of class imbalance
5. More robust validation strategy
6. Enhanced visualization and analysis
"""

import random, warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

import os

if os.environ.get("PYCHARM_HOSTED"):
    try:
        plt.switch_backend("TkAgg")
    except Exception:
        plt.switch_backend("Agg")


# =========================
# Enhanced Feature Engineering
# =========================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    df = df.copy()

    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)

    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']

    # Volatility indicators
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['atr'] = calculate_atr(df, window=14)

    # RSI
    df['rsi'] = calculate_rsi(df['close'], window=14)
    df['rsi_sma'] = df['rsi'].rolling(5).mean()

    # MACD
    df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(df['close'])

    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'] = calculate_bollinger_bands(df['close'])
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['price_volume'] = df['close'] * df['volume']

    # Momentum indicators
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['roc_5'] = df['close'].pct_change(5)
    df['roc_10'] = df['close'].pct_change(10)

    return df


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Enhanced RSI calculation"""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Use SMA for initial values, then EMA
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    # Apply EMA smoothing after initial period
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_macd(prices: pd.Series, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    bb_width = (upper_band - lower_band) / sma
    return upper_band, sma, lower_band, bb_width


def calculate_atr(df: pd.DataFrame, window=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window).mean()
    return atr


# =========================
# Enhanced Data Processing
# =========================

def create_enhanced_features(df: pd.DataFrame,
                             target_col: str = "close",
                             lookback_periods: List[int] = [1, 2, 3, 5, 10],
                             regime3: bool = False,
                             horizon: int = 1,
                             up: float = 0.02,
                             down: float = -0.02) -> pd.DataFrame:
    """Enhanced feature creation with multiple lookback periods"""

    df = df.copy()
    if "date" in df.columns:
        df = df.sort_values("date").set_index("date")

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Create lagged features
    for period in lookback_periods:
        df[f'returns_lag_{period}'] = df['returns'].shift(period)
        df[f'volatility_lag_{period}'] = df['volatility_10'].shift(period)
        df[f'rsi_lag_{period}'] = df['rsi'].shift(period)
        df[f'volume_ratio_lag_{period}'] = df['volume_ratio'].shift(period)

    # Create target variable with improved logic
    if regime3:
        # Three-class classification: Bear (0), Neutral (1), Bull (2)
        forward_returns = df[target_col].shift(-horizon) / df[target_col] - 1.0

        # Use dynamic thresholds based on rolling volatility
        rolling_vol = forward_returns.rolling(window=252, min_periods=50).std()
        dynamic_up = np.maximum(up, rolling_vol * 0.5)
        dynamic_down = np.minimum(down, -rolling_vol * 0.5)

        y = pd.Series(index=df.index, dtype="float64")
        y[forward_returns < dynamic_down] = 0  # Bear
        y[(forward_returns >= dynamic_down) & (forward_returns <= dynamic_up)] = 1  # Neutral
        y[forward_returns > dynamic_up] = 2  # Bull
        df["target"] = y
    else:
        # Binary classification with improved logic
        next_close = df[target_col].shift(-1)
        current_close = df[target_col]

        # Only create targets where both current and next prices are valid
        valid_mask = current_close.notna() & next_close.notna()

        # Use a small threshold to avoid noise
        threshold = 0.001  # 0.1% threshold
        returns = (next_close - current_close) / current_close

        df["target"] = np.nan
        df.loc[valid_mask & (returns > threshold), "target"] = 1  # Up
        df.loc[valid_mask & (returns <= -threshold), "target"] = 0  # Down
        # Ignore small movements (neutral)

    # Select features (avoid data leakage)
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'log_returns', 'price_position',
        'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
        'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20',
        'volatility_10', 'volatility_20', 'atr',
        'rsi', 'rsi_sma',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_position', 'bb_width',
        'volume_ratio', 'momentum_5', 'momentum_10',
        'roc_5', 'roc_10'
    ]

    # Add lagged features
    for period in lookback_periods:
        feature_columns.extend([
            f'returns_lag_{period}',
            f'volatility_lag_{period}',
            f'rsi_lag_{period}',
            f'volume_ratio_lag_{period}'
        ])

    # Filter existing columns
    available_features = [col for col in feature_columns if col in df.columns]

    # Create final dataset
    result_df = df[available_features + ["target"]].copy()

    # Remove rows with missing targets
    result_df = result_df[result_df["target"].notna()]

    # Handle missing feature values
    feature_cols = [c for c in result_df.columns if c != "target"]
    result_df[feature_cols] = result_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Forward fill then backward fill, finally fill with 0
    result_df[feature_cols] = result_df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0.0)

    result_df["target"] = result_df["target"].astype(int)

    return result_df


# =========================
# Enhanced CNN Architecture
# =========================

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for CNN"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class EnhancedCNN2DRegime(nn.Module):
    """Enhanced CNN with attention mechanisms and residual connections"""

    def __init__(self, n_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()

        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.channel_att1 = ChannelAttention(64)
        self.spatial_att1 = SpatialAttention()

        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        self.channel_att2 = ChannelAttention(128)
        self.spatial_att2 = SpatialAttention()

        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # Block 1 with attention
        x = self.block1(x)
        x = self.channel_att1(x)
        x = self.spatial_att1(x)

        # Block 2 with attention
        x = self.block2(x)
        x = self.channel_att2(x)
        x = self.spatial_att2(x)

        # Block 3
        x = self.block3(x)

        # Global pooling and classification
        x = self.global_pool(x)
        x = self.classifier(x)

        return x


# =========================
# Enhanced Dataset
# =========================

class EnhancedHeatmapDataset(Dataset):
    """Enhanced dataset with better sequence handling"""

    def __init__(self, X_scaled: np.ndarray, y: np.ndarray, window: int, stride: int = 1):
        self.X, self.y = self._create_sequences(X_scaled, y, window, stride)

    @staticmethod
    def _create_sequences(X_scaled: np.ndarray, y: np.ndarray, window: int, stride: int = 1):
        """Create sequences with configurable stride"""
        sequences, labels = [], []
        T = len(X_scaled)

        for i in range(window, T, stride):
            if i < len(y):  # Ensure we have a corresponding label
                sequences.append(X_scaled[i - window:i])
                labels.append(int(y[i - 1]))  # Label corresponds to the last day in the window

        if len(sequences) == 0:
            raise ValueError(f"No sequences created. Check window size ({window}) vs data length ({T})")

        sequences = np.array(sequences, dtype=np.float32)[:, None, :, :]  # Add channel dimension
        labels = np.array(labels, dtype=np.int64)

        return torch.from_numpy(sequences), torch.from_numpy(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# Enhanced Training Configuration
# =========================

@dataclass
class EnhancedTrainConfig:
    window_size: int = 30
    stride: int = 1
    scaler_type: str = "standard"  # "standard", "robust", "minmax"
    batch_size: int = 64
    epochs: int = 150
    lr: float = 1e-4
    weight_decay: float = 1e-5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    regime3: bool = False
    horizon: int = 1
    up: float = 0.015  # Reduced threshold for more balanced classes
    down: float = -0.015
    dropout_rate: float = 0.3
    use_class_weights: bool = True
    scheduler_type: str = "cosine"  # "cosine", "plateau"
    patience: int = 20


# =========================
# Enhanced Pipeline
# =========================

class EnhancedStockCNN2D:
    """Enhanced stock prediction pipeline with improved features and architecture"""

    def __init__(self, cfg: EnhancedTrainConfig):
        self.cfg = cfg
        self.model: Optional[nn.Module] = None

        # Initialize scaler
        if cfg.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif cfg.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:  # minmax
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()

        self.feature_importance_: Optional[Dict] = None

    def prepare_data(self, df: pd.DataFrame, target_col: str = "close") -> pd.DataFrame:
        """Prepare data with enhanced feature engineering"""
        return create_enhanced_features(
            df,
            target_col=target_col,
            regime3=self.cfg.regime3,
            horizon=self.cfg.horizon,
            up=self.cfg.up,
            down=self.cfg.down
        )

    def analyze_data_quality(self, df_clean: pd.DataFrame):
        """Analyze data quality and distribution"""
        print("\n=== Data Quality Analysis ===")
        print(f"Total samples: {len(df_clean)}")
        print(f"Features: {len([c for c in df_clean.columns if c != 'target'])}")

        # Target distribution
        target_dist = df_clean['target'].value_counts().sort_index()
        print(f"Target distribution: {dict(target_dist)}")

        # Missing values
        missing_pct = (df_clean.isnull().sum() / len(df_clean) * 100).round(2)
        if missing_pct.sum() > 0:
            print("Features with missing values:")
            print(missing_pct[missing_pct > 0])

        # Feature statistics
        feature_cols = [c for c in df_clean.columns if c != 'target']
        print(f"\nFeature statistics summary:")
        print(
            f"Mean absolute correlation with target: {abs(df_clean[feature_cols].corrwith(df_clean['target'])).mean():.3f}")

        return target_dist

    def create_splits(self, df_clean: pd.DataFrame):
        """Create time-based train/val/test splits"""
        n = len(df_clean)
        train_end = int(n * self.cfg.train_ratio)
        val_end = int(n * (self.cfg.train_ratio + self.cfg.val_ratio))

        train_mask = np.arange(n) < train_end
        val_mask = (np.arange(n) >= train_end) & (np.arange(n) < val_end)
        test_mask = np.arange(n) >= val_end

        return train_mask, val_mask, test_mask

    def fit(self, df_clean: pd.DataFrame):
        """Enhanced training pipeline"""

        # Set random seed
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        if len(df_clean) == 0:
            raise ValueError("Empty dataframe provided")

        # Analyze data quality
        target_dist = self.analyze_data_quality(df_clean)

        # Prepare features and targets
        feature_cols = [c for c in df_clean.columns if c != 'target']
        X = df_clean[feature_cols].values.astype(np.float32)
        y = df_clean['target'].values.astype(int)

        # Create time-based splits
        train_mask, val_mask, test_mask = self.create_splits(df_clean)

        # Fit scaler only on training data
        self.scaler.fit(X[train_mask])
        X_train_scaled = self.scaler.transform(X[train_mask])
        X_val_scaled = self.scaler.transform(X[val_mask])
        X_test_scaled = self.scaler.transform(X[test_mask])

        # Create datasets
        train_dataset = EnhancedHeatmapDataset(
            X_train_scaled, y[train_mask], self.cfg.window_size, self.cfg.stride
        )
        val_dataset = EnhancedHeatmapDataset(
            X_val_scaled, y[val_mask], self.cfg.window_size, 1  # No stride for validation
        )
        test_dataset = EnhancedHeatmapDataset(
            X_test_scaled, y[test_mask], self.cfg.window_size, 1  # No stride for testing
        )

        print(f"\nDataset sizes after windowing:")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.cfg.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.cfg.batch_size, shuffle=False
        )

        # Initialize model
        n_classes = len(np.unique(y))
        self.model = EnhancedCNN2DRegime(n_classes=n_classes, dropout_rate=self.cfg.dropout_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.model.to(device)

        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        if self.cfg.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.epochs, eta_min=1e-6)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # Setup loss function with class weights
        if self.cfg.use_class_weights:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_dataset.y.numpy()),
                y=train_dataset.y.numpy()
            )
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop with enhanced monitoring
        history = self._train_model(
            self.model, train_loader, val_loader, optimizer, scheduler, criterion, device
        )

        # Final evaluation
        test_results = self._evaluate_model(test_loader, device)

        return history, test_results

    def _train_model(self, model, train_loader, val_loader, optimizer, scheduler, criterion, device):
        """Enhanced training loop with better monitoring"""

        history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'lr': [], 'best_val_loss': float('inf'), 'best_epoch': 0
        }

        best_model_state = None
        patience_counter = 0

        print(f"\nStarting training for {self.cfg.epochs} epochs...")

        for epoch in range(1, self.cfg.epochs + 1):
            # Training phase
            model.train()
            train_loss = train_acc = train_samples = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                train_acc += pred.eq(target.view_as(pred)).sum().item()
                train_samples += data.size(0)

            train_loss /= train_samples
            train_acc /= train_samples

            # Validation phase
            model.eval()
            val_loss = val_acc = val_samples = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_acc += pred.eq(target.view_as(pred)).sum().item()
                    val_samples += data.size(0)

            val_loss /= val_samples
            val_acc /= val_samples

            # Update scheduler
            if self.cfg.scheduler_type == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)

            # Early stopping and model saving
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if epoch % 10 == 0 or epoch <= 5:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.2e}")

            # Early stopping
            if patience_counter >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model from epoch {history['best_epoch']}")

        return history

    def _evaluate_model(self, test_loader, device):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')

        print("\n=== Test Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-macro: {f1_macro:.4f}")
        print(f"F1-weighted: {f1_weighted:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data"""
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        # Scale the input
        X_scaled = self.scaler.transform(X)

        # Create sequences
        window = self.cfg.window_size
        sequences = []

        for i in range(window, len(X_scaled)):
            sequences.append(X_scaled[i - window:i])

        if len(sequences) == 0:
            raise ValueError("Not enough data to create sequences")

        sequences = np.array(sequences)[:, None, :, :]  # Add channel dimension
        sequences = torch.from_numpy(sequences).float().to(device)

        # Predict in batches
        probabilities = []
        batch_size = self.cfg.batch_size

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            output = self.model(batch)
            probs = torch.softmax(output, dim=1)
            probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)


# =========================
# Enhanced Visualization
# =========================

def plot_enhanced_history(history: dict, save_path: str = None):
    """Enhanced training history visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.8)
    axes[0, 0].axvline(x=history['best_epoch'], color='red', linestyle='--', alpha=0.5, label='Best Model')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Acc', alpha=0.8)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', alpha=0.8)
    axes[0, 1].axvline(x=history['best_epoch'], color='red', linestyle='--', alpha=0.5, label='Best Model')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate plot
    axes[1, 0].plot(history['lr'], label='Learning Rate', color='orange', alpha=0.8)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].plot(loss_diff, label='Val - Train Loss', color='red', alpha=0.8)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_title('Overfitting Indicator (Val - Train Loss)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_analysis(test_results: dict, df_clean: pd.DataFrame = None):
    """Analyze prediction results"""
    targets = test_results['targets']
    predictions = test_results['predictions']
    probabilities = test_results['probabilities']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Confusion matrix heatmap
    cm = confusion_matrix(targets, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')

    # Prediction confidence distribution
    max_probs = np.max(probabilities, axis=1)
    axes[0, 1].hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Prediction Confidence Distribution')
    axes[0, 1].set_xlabel('Max Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # Class probability distributions
    n_classes = probabilities.shape[1]
    for i in range(n_classes):
        axes[1, 0].hist(probabilities[:, i], bins=20, alpha=0.5, label=f'Class {i}')
    axes[1, 0].set_title('Class Probability Distributions')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Prediction accuracy by confidence
    confidence_bins = np.linspace(0.5, 1.0, 11)
    accuracies = []

    for i in range(len(confidence_bins) - 1):
        mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            acc = accuracy_score(targets[mask], predictions[mask])
            accuracies.append(acc)
        else:
            accuracies.append(np.nan)

    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    axes[1, 1].plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=6)
    axes[1, 1].set_title('Accuracy vs Prediction Confidence')
    axes[1, 1].set_xlabel('Confidence Bin')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.show()


# =========================
# Data Loading Functions (keep original ones)
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert string numbers (with commas, scientific notation) to numeric"""
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def adapt_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt column names from different data sources"""
    df = df.copy()

    def norm(s: str) -> str:
        return (s.strip().lower()
                .replace(" ", "")
                .replace("_", "")
                .replace("(", "")
                .replace(")", "")
                .replace("%", "")
                .replace("-", "")
                .replace("／", "/"))

    df.columns = [norm(c) for c in df.columns]

    alias = {
        "tradedate": "date", "date": "date",
        "secucode": "asset", "stockcode": "asset", "tscode": "asset",
        "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
        "transactionamount": "turnover_value", "turnovervalue": "turnover_value",
        "amplitude": "amplitude",
        "turnoverrate": "turnover_rate",
        "changepercentage": "chg_pct", "changepercent": "chg_pct",
        "changeamount": "chg_amt"
    }
    df.columns = [alias.get(c, c) for c in df.columns]

    required = ["date", "open", "high", "low", "close", "volume"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # Enhanced date cleaning
    raw_date = (
        df["date"].astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u200b", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace("／", "/", regex=False)
        .str.replace("年", "/", regex=False)
        .str.replace("月", "/", regex=False)
        .str.replace("日", "", regex=False)
        .str.replace(".", "/", regex=False)
        .str.replace("-", "/", regex=False)
        .str.strip()
    )

    date_parsed = pd.to_datetime(raw_date, errors="coerce", format="%m/%d/%Y")
    mask = date_parsed.isna()
    if mask.any():
        date_parsed.loc[mask] = pd.to_datetime(raw_date[mask], errors="coerce", format="%Y/%m/%d")
        mask = date_parsed.isna()
    if mask.any():
        date_parsed.loc[mask] = pd.to_datetime(raw_date[mask], errors="coerce", format="%d/%m/%Y")
        mask = date_parsed.isna()
    if mask.any():
        date_parsed.loc[mask] = pd.to_datetime(raw_date[mask], errors="coerce", infer_datetime_format=True)

    df["date"] = date_parsed
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume",
                    "turnover_value", "amplitude", "turnover_rate", "chg_pct", "chg_amt"]
    df = _coerce_numeric(df, [c for c in numeric_cols if c in df.columns])

    for c in ["turnover_value", "amplitude", "turnover_rate", "chg_pct", "chg_amt"]:
        if c in df.columns and df[c].isna().all():
            df.drop(columns=[c], inplace=True)

    return df


def load_generic_market_csv(file_path: str, asset_code: Optional[str] = None) -> pd.DataFrame:
    """Load and process market data CSV"""
    df = pd.read_csv(file_path, dtype=str)
    df = adapt_columns(df)
    if asset_code is not None and "asset" in df.columns:
        df = df[df["asset"].astype(str) == str(asset_code)].copy()
    return df


# =========================
# Main execution
# =========================

def main():
    """Main execution function with enhanced pipeline"""
    set_seed(42)

    # Configuration
    file_path = r"F:\VesperSet\stock_data_analysis\data\raw\indexData\train_HS300_daily_kline.csv"
    asset_code = None  # Set to specific asset code if needed

    # Enhanced configuration
    config = EnhancedTrainConfig(
        window_size=30,  # Longer window for more context
        stride=1,  # No stride for maximum data usage
        scaler_type="standard",  # StandardScaler often works better
        batch_size=64,  # Smaller batch size for better generalization
        epochs=150,  # More epochs with early stopping
        lr=1e-4,  # Lower learning rate for stability
        weight_decay=1e-5,  # Light regularization
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        regime3=False,  # Binary classification
        horizon=1,
        up=0.01,  # 1% threshold
        down=-0.01,  # -1% threshold
        dropout_rate=0.3,
        use_class_weights=True,  # Handle class imbalance
        scheduler_type="cosine",  # Cosine annealing scheduler
        patience=20  # Early stopping patience
    )

    print("Loading and processing data...")
    df = load_generic_market_csv(file_path, asset_code=asset_code)

    print(f"Raw data shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head().to_string(index=False))

    # Initialize enhanced pipeline
    pipeline = EnhancedStockCNN2D(config)

    print("\nPreparing features...")
    df_clean = pipeline.prepare_data(df, target_col="close")

    if len(df_clean) == 0:
        print("ERROR: No data after preprocessing!")
        return

    print(f"Processed data shape: {df_clean.shape}")

    # Train the model
    print("\nStarting training...")
    history, test_results = pipeline.fit(df_clean)

    # Visualize results
    print("\nGenerating visualizations...")
    plot_enhanced_history(history)
    plot_prediction_analysis(test_results, df_clean)

    # Additional analysis
    print(f"\nBest validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")

    # Feature importance analysis (simple version based on correlation)
    feature_cols = [c for c in df_clean.columns if c != 'target']
    correlations = df_clean[feature_cols].corrwith(df_clean['target']).abs().sort_values(ascending=False)

    print("\nTop 10 most correlated features:")
    for i, (feature, corr) in enumerate(correlations.head(10).items()):
        print(f"{i + 1:2d}. {feature}: {corr:.4f}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()