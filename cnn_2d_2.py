# -*- coding: utf-8 -*-
import os
import gc
import random
import warnings
import platform
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    classification_report, balanced_accuracy_score, f1_score,
    confusion_matrix, matthews_corrcoef, average_precision_score,
    precision_recall_curve, roc_auc_score, log_loss
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# IMPROVED CONFIGURATION
# =========================

@dataclass
class ImprovedModelConfig:
    """Improved model configuration with residual connections"""
    # Architecture type
    model_type: str = "improved_cnn"

    # CNN parameters
    base_channels: int = 16  # Increased from 8
    channel_multiplier: float = 2.0
    num_blocks: int = 3  # Increased from 2
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])

    # Attention parameters
    use_attention: bool = True  # Enabled
    attention_type: str = "se"

    # Regularization
    dropout_rate: float = 0.2  # Reduced from 0.3
    dropblock_rate: float = 0.1
    use_dropout2d: bool = True

    # Normalization
    norm_type: str = "batch"

    # Activation
    activation: str = "relu"

    # New parameters for improved architecture
    use_residual: bool = True
    use_multi_scale_pooling: bool = True

@dataclass
class ImprovedTrainingConfig:
    """Improved training configuration for better convergence"""
    # Basic training parameters
    batch_size: int = 64  # Increased from 32
    epochs: int = 200  # Increased from 100
    early_stopping_patience: int = 30  # Increased from 15

    # Optimizer parameters
    optimizer_type: str = "adamw"
    learning_rate: float = 2e-4  # Decreased from 5e-4
    weight_decay: float = 1e-5  # Decreased from 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler parameters
    scheduler_type: str = "cosine_restart"  # Changed from reduce_on_plateau
    warmup_epochs: int = 15
    T_0: int = 40  # Cosine restart period
    T_mult: int = 1
    eta_min: float = 1e-6

    # Loss function
    loss_type: str = "focal"  # Changed from weighted_ce
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # Mixed precision and optimization
    use_amp: bool = True
    gradient_clip_val: float = 0.5  # Reduced from 1.0
    accumulate_grad_batches: int = 1

    # Monitoring
    monitor_metric: str = "val_f1_macro"
    monitor_mode: str = "max"

@dataclass
class ImprovedDataConfig:
    """Improved data processing configuration"""
    # Data parameters
    window_size: int = 40  # Increased from 30
    stride: int = 1
    prediction_horizon: int = 1

    # Feature engineering
    feature_engineering_level: str = "enhanced"  # Upgraded from basic
    max_features: int = 20  # Increased from 15
    feature_selection_method: str = "f_classif"
    correlation_threshold: float = 0.95

    # Improved labeling strategy
    labeling_method: str = "adaptive_percentile"  # Changed from simple_threshold
    binary_threshold: float = 0.005  # Reduced from 0.01
    volatility_window: int = 60  # Increased from 20
    percentile_window: int = 100  # New parameter
    up_percentile: float = 0.65  # New parameter
    down_percentile: float = 0.35  # New parameter

    # Scaling
    scaler_type: str = "robust"
    feature_wise_normalization: bool = False
    use_global_scaler: bool = True

    # Time series validation
    validation_method: str = "walk_forward"
    n_splits: int = 3
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    embargo_period: int = 5

@dataclass
class ImprovedConfig:
    """Complete improved configuration"""
    model: ImprovedModelConfig = field(default_factory=ImprovedModelConfig)
    training: ImprovedTrainingConfig = field(default_factory=ImprovedTrainingConfig)
    data: ImprovedDataConfig = field(default_factory=ImprovedDataConfig)

    # System settings
    device: str = "auto"
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 42

    # Logging and checkpoints
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # Experiment tracking
    use_tensorboard: bool = True
    experiment_name: str = "improved_stock_cnn"

# =========================
# FOCAL LOSS IMPLEMENTATION
# =========================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =========================
# SQUEEZE-AND-EXCITATION BLOCK
# =========================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for attention"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# =========================
# RESIDUAL BLOCK
# =========================

class ResidualBlock(nn.Module):
    """Residual block with optional SE attention"""
    
    def __init__(self, in_channels, out_channels, dropout_rate, use_attention=False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Optional attention
        self.se_block = SEBlock(out_channels) if use_attention else None
        
        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        if self.se_block is not None:
            out = self.se_block(out)
        
        out += residual
        out = self.relu(out)
        
        return out

# =========================
# IMPROVED MODEL ARCHITECTURE
# =========================

class ImprovedStockCNN(nn.Module):
    """Improved CNN with residual connections and attention"""
    
    def __init__(self, num_classes: int, config: ImprovedModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, config.base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        in_channels = config.base_channels
        
        for i in range(config.num_blocks):
            out_channels = int(config.base_channels * (config.channel_multiplier ** i))
            
            if config.use_residual:
                block = ResidualBlock(
                    in_channels, out_channels, 
                    config.dropout_rate, 
                    use_attention=config.use_attention
                )
            else:
                # Fallback to simple conv block
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(config.dropout_rate)
                )
            
            self.res_blocks.append(block)
            in_channels = out_channels
        
        # Multi-scale pooling
        if config.use_multi_scale_pooling:
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.global_max_pool = nn.AdaptiveMaxPool2d(1)
            classifier_input = in_channels * 2
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            classifier_input = in_channels
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(classifier_input, classifier_input),
            nn.BatchNorm1d(classifier_input),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(classifier_input, classifier_input // 2),
            nn.BatchNorm1d(classifier_input // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(classifier_input // 2, num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input processing
        x = self.input_conv(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Multi-scale pooling
        if self.config.use_multi_scale_pooling:
            avg_pool = self.global_avg_pool(x).flatten(1)
            max_pool = self.global_max_pool(x).flatten(1)
            x = torch.cat([avg_pool, max_pool], dim=1)
        else:
            x = self.global_pool(x).flatten(1)
        
        # Classification
        x = self.classifier(x)
        
        return x

    def _initialize_weights(self):
        """Xavier initialization for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:            # <— add this guard
                    nn.init.constant_(m.bias, 0)

# =========================
# ENHANCED FEATURE ENGINEERING
# =========================

class EnhancedFeatureEngineer:
    """Enhanced feature engineering with more indicators"""

    def __init__(self, config: ImprovedDataConfig):
        self.config = config
        self.column_mapping = {}

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering pipeline"""
        logger.info("Starting enhanced feature engineering...")
        df = df.copy()

        self.column_mapping = self._detect_columns(df)

        # Enhanced technical indicators
        df = self._add_enhanced_indicators(df)
        
        # Remove date column if present
        date_col = self.column_mapping.get('date')
        if date_col and date_col in df.columns:
            df = df.drop(columns=[date_col])

        # Clean data
        try:
            df = df.ffill().bfill().fillna(0.0)
        except AttributeError:
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        # Remove non-numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df[col] = 0.0
        
        df = df.fillna(0.0)
        
        # Remove datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
        if len(datetime_cols) > 0:
            logger.warning(f"Removing datetime columns: {datetime_cols.tolist()}")
            df = df.drop(columns=datetime_cols)

        logger.info(f"Enhanced feature engineering completed. Total features: {len(df.columns)}")
        return df

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping = {}
        columns = {col.lower(): col for col in df.columns}
        
        patterns = {
            'open': ['open'],
            'high': ['high'],
            'low': ['low'],
            'close': ['close'],
            'volume': ['volume', 'vol'],
            'date': ['date', 'time', 'datetime']
        }
        
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                for col_lower, col_original in columns.items():
                    if pattern in col_lower:
                        mapping[key] = col_original
                        break
                if key in mapping:
                    break
        
        return mapping

    def _add_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        close = df[self.column_mapping.get('close', 'Close')]
        high = df[self.column_mapping.get('high', 'High')]
        low = df[self.column_mapping.get('low', 'Low')]
        volume = df[self.column_mapping.get('volume', 'Volume')]

        # Price-based indicators
        df['returns'] = close.pct_change()
        df['log_returns'] = np.log(close / close.shift(1))
        
        # Multiple timeframe moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = close.rolling(window=period).mean()
            df[f'ema_{period}'] = close.ewm(span=period).mean()
            df[f'price_sma_ratio_{period}'] = close / df[f'sma_{period}']
            df[f'price_ema_ratio_{period}'] = close / df[f'ema_{period}']

        # Volatility indicators
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            df[f'atr_{period}'] = self._calculate_atr(high, low, close, period)

        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(close, 14)
        df['rsi_21'] = self._calculate_rsi(close, 21)
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        bb_period = 20
        bb_sma = close.rolling(window=bb_period).mean()
        bb_std = close.rolling(window=bb_period).std()
        df['bb_upper'] = bb_sma + (bb_std * 2)
        df['bb_lower'] = bb_sma - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_sma
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume indicators
        df['volume_sma_10'] = volume.rolling(window=10).mean()
        df['volume_ratio'] = volume / df['volume_sma_10']
        df['vwap'] = (close * volume).cumsum() / volume.cumsum()
        df['price_vwap_ratio'] = close / df['vwap']

        # Price position indicators
        df['hl_ratio'] = (high - low) / close
        df['price_position'] = (close - low) / (high - low + 1e-8)
        
        # Gap indicators
        df['gap'] = (df[self.column_mapping.get('open', 'Open')] - close.shift(1)) / close.shift(1)

        return df

    def _calculate_rsi(self, prices, period):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, high, low, close, period):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

# =========================
# IMPROVED LABEL GENERATOR
# =========================

class ImprovedLabelGenerator:
    """Improved label generation with adaptive thresholding"""
    
    def __init__(self, config: ImprovedDataConfig):
        self.config = config

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate balanced labels using adaptive thresholding"""
        df = df.copy()
        
        close_col = self._get_close_column(df)
        
        # Forward returns
        forward_returns = df[close_col].shift(-self.config.prediction_horizon) / df[close_col] - 1.0
        
        # Adaptive percentile-based thresholds
        up_threshold = forward_returns.rolling(
            window=self.config.percentile_window, 
            min_periods=20
        ).quantile(self.config.up_percentile)
        
        down_threshold = forward_returns.rolling(
            window=self.config.percentile_window, 
            min_periods=20
        ).quantile(self.config.down_percentile)
        
        # Fill NaN with fallback values
        up_threshold = up_threshold.fillna(self.config.binary_threshold)
        down_threshold = down_threshold.fillna(-self.config.binary_threshold)
        
        # Create initial labels
        conditions = [
            forward_returns > up_threshold,
            forward_returns < down_threshold
        ]
        choices = [2, 0]  # Up, Down
        df['target'] = np.select(conditions, choices, default=1)  # Neutral
        
        # Ensure better class balance
        df = self._ensure_class_balance(df, forward_returns)
        
        # Log class distribution
        class_counts = df['target'].value_counts().sort_index()
        total = len(df)
        logger.info(f"Improved label distribution:")
        for class_id, count in class_counts.items():
            class_name = ['Down', 'Neutral', 'Up'][class_id]
            logger.info(f"{class_name} ({class_id}): {count} ({count/total*100:.1f}%)")
        
        return df

    def _ensure_class_balance(self, df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Ensure minimum class representation"""
        target_min_pct = 0.20  # Minimum 20% for each class
        n_samples = len(df)
        min_samples = int(n_samples * target_min_pct)
        
        class_counts = df['target'].value_counts()
        
        # Adjust thresholds if classes are under-represented
        for class_id in [0, 2]:  # Don't adjust neutral
            current_count = class_counts.get(class_id, 0)
            if current_count < min_samples:
                needed = min_samples - current_count
                neutral_mask = df['target'] == 1
                neutral_returns = returns[neutral_mask]
                
                if len(neutral_returns) > needed:
                    if class_id == 0:  # Down class
                        # Convert lowest neutral returns to down
                        threshold = neutral_returns.quantile(needed / len(neutral_returns))
                        convert_mask = neutral_mask & (returns <= threshold)
                    else:  # Up class
                        # Convert highest neutral returns to up
                        threshold = neutral_returns.quantile(1 - needed / len(neutral_returns))
                        convert_mask = neutral_mask & (returns >= threshold)
                    
                    df.loc[convert_mask, 'target'] = class_id
        
        return df

    def _get_close_column(self, df: pd.DataFrame) -> str:
        for col in ['Close', 'close', 'CLOSE']:
            if col in df.columns:
                return col
        raise ValueError("No close column found")

# =========================
# IMPROVED DATA PROCESSOR
# =========================

class ImprovedDataProcessor:
    """Improved data processor with enhanced features"""
    
    def __init__(self, config: ImprovedDataConfig):
        self.config = config

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_engineer = EnhancedFeatureEngineer(self.config)
        df = feature_engineer.engineer_features(df)
        
        label_generator = ImprovedLabelGenerator(self.config)
        df = label_generator.generate_labels(df)
        
        return df

# =========================
# IMPROVED DATASET CLASS
# =========================

class ImprovedStockDataset(Dataset):
    """Improved dataset with better sequence handling"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, window_size: int, 
                 stride: int = 1, scaler=None, fit_scaler: bool = True):
        self.window_size = window_size
        self.stride = stride
        self.scaler = scaler
        
        # Fit scaler if needed
        if fit_scaler and self.scaler is not None:
            self.scaler.fit(data)
        
        # Scale data
        if self.scaler is not None:
            data = self.scaler.transform(data)
        
        self.sequences, self.targets = self._create_sequences(data, labels)
        self.sequences = self.sequences.astype(np.float32)
        self.targets = self.targets.astype(np.int64)
        
        logger.info(f"Dataset created: {len(self.sequences)} samples, "
                   f"sequence shape: {self.sequences.shape[1:]}")

    def _create_sequences(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with improved handling"""
        n_samples, n_features = data.shape
        
        sequences = []
        targets = []
        
        for i in range(0, n_samples - self.window_size + 1, self.stride):
            if i + self.window_size <= n_samples:
                seq = data[i:i + self.window_size]
                target = labels[i + self.window_size - 1]
                
                # Skip sequences with invalid targets
                if not np.isnan(target):
                    sequences.append(seq)
                    targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Reshape for CNN: (batch_size, channels, height, width)
        sequences = sequences[:, None, :, :]
        
        return sequences, targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        return torch.from_numpy(sequence.copy()), torch.tensor(target, dtype=torch.long)

# =========================
# IMPROVED TRAINER CLASS
# =========================

class ImprovedTrainer:
    """Improved trainer with better loss functions and monitoring"""
    
    def __init__(self, config: ImprovedConfig):
        self.config = config
        if platform.system().lower().startswith('win'):
            self.config.num_workers = 0
        
        self.device = self._get_device()
        self.writer = SummaryWriter(Path(config.log_dir) / config.experiment_name) if config.use_tensorboard else None
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = GradScaler() if config.training.use_amp else None
        
        self.best_score = 0.0
        self.patience_counter = 0

    def fit(self, train_data, train_labels, val_data, val_labels):
        """Train the model with improved handling"""
        
        # Create scalers
        train_scaler = RobustScaler() if self.config.data.scaler_type == "robust" else StandardScaler()
        
        # Create datasets
        train_dataset = ImprovedStockDataset(
            train_data, train_labels, self.config.data.window_size, 
            self.config.data.stride, train_scaler, fit_scaler=True
        )
        
        val_dataset = ImprovedStockDataset(
            val_data, val_labels, self.config.data.window_size,
            stride=1, scaler=train_scaler, fit_scaler=False
        )
        
        # Create data loaders
        train_loader = self._create_balanced_dataloader(train_dataset, shuffle=True)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Setup training
        self._setup_training(train_labels)
        
        # Training loop
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate_epoch(val_loader)
            
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                elif isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                    self.scheduler.step()
            
            # Early stopping check
            current_score = val_metrics[self.config.training.monitor_metric]
            if current_score > self.best_score:
                self.best_score = current_score
                self.patience_counter = 0
                self._save_checkpoint('best_model.pth')
                logger.info(f"New best score: {self.best_score:.4f}")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break
        
        if self.writer:
            self.writer.close()
        
        return {'best_score': self.best_score}

    def _create_balanced_dataloader(self, dataset, shuffle=True):
        """Create balanced dataloader with improved sampling"""
        
        # Get labels from dataset
        dataset_labels = dataset.targets
        
        # Calculate class weights
        unique_classes, class_counts = np.unique(dataset_labels, return_counts=True)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / np.sum(class_weights) * len(class_weights)
        
        # Create sample weights
        sample_weights = class_weights[dataset_labels]
        
        # Create weighted sampler
        if shuffle:
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        else:
            sampler = None
        
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    def _setup_training(self, train_labels):
        """Setup model, optimizer, scheduler, and loss function"""
        
        num_classes = len(np.unique(train_labels))
        logger.info(f"Setting up training for {num_classes} classes")
        
        # Model
        self.model = ImprovedStockCNN(num_classes, self.config.model).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer
        if self.config.training.optimizer_type == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas
            )
        elif self.config.training.optimizer_type == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        
        # Scheduler
        if self.config.training.scheduler_type == "cosine_restart":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.training.T_0,
                T_mult=self.config.training.T_mult,
                eta_min=self.config.training.eta_min
            )
        elif self.config.training.scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=8
            )
        
        # Loss function
        if self.config.training.loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=self.config.training.focal_alpha,
                gamma=self.config.training.focal_gamma
            )
            logger.info(f"Using Focal Loss with alpha={self.config.training.focal_alpha}, gamma={self.config.training.focal_gamma}")
        else:
            # Weighted cross entropy as fallback
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using weighted cross-entropy with weights: {class_weights}")

    def _train_epoch(self, train_loader):
        """Training epoch with improved monitoring"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.training.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.training.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_val
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_val
                )
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = balanced_accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_per_class = f1_score(all_targets, all_preds, average=None)
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_f1_macro': f1_macro,
            'train_f1_per_class': f1_per_class
        }

    def _validate_epoch(self, val_loader):
        """Validation epoch with comprehensive metrics"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                with autocast(enabled=self.config.training.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                
                preds = output.argmax(dim=1)
                probs = F.softmax(output, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = balanced_accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_per_class = f1_score(all_targets, all_preds, average=None)
        mcc = matthews_corrcoef(all_targets, all_preds)
        
        # AUC (if possible)
        try:
            auc = roc_auc_score(all_targets, np.array(all_probs), multi_class='ovr')
        except ValueError:
            auc = 0.5
        
        # Class distribution
        unique, counts = np.unique(all_targets, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_f1_macro': f1_macro,
            'val_f1_per_class': f1_per_class,
            'val_mcc': mcc,
            'val_auc': auc,
            'val_class_distribution': class_dist
        }

    def _log_metrics(self, train_metrics, val_metrics, epoch):
        """Log comprehensive training metrics"""
        # Log per-class F1 scores
        train_f1_per_class = train_metrics.get('train_f1_per_class', [])
        val_f1_per_class = val_metrics.get('val_f1_per_class', [])
        
        class_names = ['Down', 'Neutral', 'Up']
        per_class_str = ""
        for i, (train_f1, val_f1) in enumerate(zip(train_f1_per_class, val_f1_per_class)):
            if i < len(class_names):
                per_class_str += f"{class_names[i]}: {train_f1:.3f}/{val_f1:.3f} "
        
        log_str = (
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val F1 Macro: {val_metrics['val_f1_macro']:.4f} | "
            f"Val MCC: {val_metrics['val_mcc']:.4f} | "
            f"Per-class F1 (Train/Val): {per_class_str}"
        )
        logger.info(log_str)
        
        if self.writer:
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Training/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Validation/{key}', value, epoch)
            
            # Log per-class F1 scores
            for i, f1 in enumerate(train_f1_per_class):
                if i < len(class_names):
                    self.writer.add_scalar(f'Training/F1_{class_names[i]}', f1, epoch)
            
            for i, f1 in enumerate(val_f1_per_class):
                if i < len(class_names):
                    self.writer.add_scalar(f'Validation/F1_{class_names[i]}', f1, epoch)
            
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

    def _save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
            'config': self.config
        }
        
        torch.save(state, checkpoint_dir / filename)
        logger.info(f"Checkpoint saved to {checkpoint_dir / filename}")

    def _get_device(self):
        """Get appropriate device for training"""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            return device
        else:
            return torch.device(self.config.device)

# =========================
# IMPROVED TIME SERIES CV
# =========================

class ImprovedTimeSeriesCV:
    """Improved time series cross-validation with better splits"""
    
    def __init__(self, config: ImprovedDataConfig):
        self.config = config

    def split(self, data: pd.DataFrame):
        """Create improved time series splits"""
        n = len(data)
        min_train_size = max(200, self.config.window_size * 3)  # Increased minimum
        
        if n < min_train_size:
            logger.warning(f"Dataset too small ({n} samples) for time series CV")
            return []
        
        # Calculate split sizes
        train_size = max(min_train_size, int(n * self.config.train_ratio))
        val_size = max(100, int(n * self.config.val_ratio))
        
        splits = []
        
        # Walk-forward validation with larger steps
        step_size = max(50, val_size // 3)  # Larger steps between splits
        
        for i in range(self.config.n_splits):
            # Calculate split positions
            split_offset = i * step_size
            
            train_start = split_offset
            train_end = train_start + train_size
            
            val_start = train_end + self.config.embargo_period
            val_end = val_start + val_size
            
            # Check if we have enough data
            if val_end <= n:
                train_indices = np.arange(train_start, train_end)
                val_indices = np.arange(val_start, val_end)
                
                logger.info(f"Split {i+1}: Train[{train_start}:{train_end}] ({len(train_indices)} samples), "
                           f"Val[{val_start}:{val_end}] ({len(val_indices)} samples)")
                splits.append((train_indices, val_indices))
            else:
                logger.warning(f"Split {i+1} exceeds data length, skipping")
                break
        
        # Ensure at least one split
        if not splits:
            train_end = int(n * 0.75)
            val_start = train_end + self.config.embargo_period
            if val_start < n:
                train_indices = np.arange(0, train_end)
                val_indices = np.arange(val_start, n)
                splits.append((train_indices, val_indices))
                logger.info("Using single fallback split")
        
        return splits

# =========================
# IMPROVED MAIN PREDICTOR CLASS
# =========================

class ImprovedStockPredictor:
    """Improved stock predictor with comprehensive enhancements"""
    
    def __init__(self, config: ImprovedConfig):
        self.config = config
        self.data_processor = ImprovedDataProcessor(config.data)
        self.cv = ImprovedTimeSeriesCV(config.data)
        self._set_seeds()
        
        # Create directories
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def fit(self, df: pd.DataFrame):
        """Fit the model with improved pipeline"""
        logger.info("Starting improved model training pipeline")
        
        # Process data
        df_processed = self.data_processor.fit_transform(df)
        
        # Remove rows with NaN targets
        df_processed = df_processed.dropna(subset=['target'])
        
        if len(df_processed) < 200:
            raise ValueError(f"Insufficient data after processing: {len(df_processed)} samples")
        
        # Prepare features and targets
        feature_cols = [col for col in df_processed.columns if col != 'target']
        X = df_processed[feature_cols].values
        y = df_processed['target'].values
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        # Cross-validation
        cv_results = []
        splits = self.cv.split(df_processed)
        
        if not splits:
            raise ValueError("No valid CV splits generated. Check data size and configuration.")
        
        logger.info(f"Starting {len(splits)}-fold cross-validation")
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training fold {fold + 1}/{len(splits)}")
            logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            
            # Split data
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Check class distribution
            train_dist = np.bincount(y_train)
            val_dist = np.bincount(y_val)
            logger.info(f"Train class distribution: {train_dist}")
            logger.info(f"Val class distribution: {val_dist}")
            
            # Feature selection if needed
            if X_train.shape[1] > self.config.data.max_features:
                logger.info(f"Selecting top {self.config.data.max_features} features")
                selector = SelectKBest(
                    f_classif if self.config.data.feature_selection_method == "f_classif" else mutual_info_classif,
                    k=self.config.data.max_features
                )
                X_train = selector.fit_transform(X_train, y_train)
                X_val = selector.transform(X_val)
                logger.info(f"Feature selection completed. New shape: {X_train.shape}")
            
            # Train model
            try:
                trainer = ImprovedTrainer(self.config)
                fold_results = trainer.fit(X_train, y_train, X_val, y_val)
                cv_results.append(fold_results)
                logger.info(f"Fold {fold + 1} completed. Best score: {fold_results['best_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Training failed for fold {fold + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                cv_results.append({'best_score': 0.0})
                
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Log final results
        self._log_cv_results(cv_results)
        
        return {
            'cv_results': cv_results,
            'config': self.config,
            'mean_score': np.mean([r['best_score'] for r in cv_results]),
            'std_score': np.std([r['best_score'] for r in cv_results])
        }

    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Random seeds set to {self.config.seed}")

    def _log_cv_results(self, cv_results):
        """Log comprehensive cross-validation results"""
        scores = [res['best_score'] for res in cv_results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        logger.info("\n" + "="*70)
        logger.info("IMPROVED CROSS-VALIDATION RESULTS")
        logger.info("="*70)
        logger.info(f"Mean CV Score: {mean_score:.4f} ± {std_score:.4f}")
        logger.info(f"Individual fold scores: {[f'{s:.4f}' for s in scores]}")
        logger.info(f"Best single fold: {max(scores):.4f}")
        logger.info(f"Worst single fold: {min(scores):.4f}")
        logger.info(f"Score range: {max(scores) - min(scores):.4f}")
        
        if len(scores) > 1 and std_score > 0:
            cv_coefficient = std_score / mean_score
            logger.info(f"Coefficient of Variation: {cv_coefficient:.4f}")
            
            # Stability assessment
            if cv_coefficient < 0.1:
                stability = "Very Stable"
            elif cv_coefficient < 0.2:
                stability = "Stable"
            elif cv_coefficient < 0.3:
                stability = "Moderately Stable"
            else:
                stability = "Unstable"
            
            logger.info(f"Model Stability: {stability}")
        
        # Performance assessment
        if mean_score > 0.4:
            performance = "Excellent"
        elif mean_score > 0.3:
            performance = "Good"
        elif mean_score > 0.2:
            performance = "Fair"
        elif mean_score > 0.1:
            performance = "Poor"
        else:
            performance = "Very Poor"
        
        logger.info(f"Overall Performance: {performance}")
        logger.info("="*70)

# =========================
# MAIN EXECUTION FUNCTION
# =========================

def main():
    """Main execution function with improved error handling"""
    logger.info("Starting Improved Stock CNN Pipeline")
    
    # Configuration
    config = ImprovedConfig()
    logger.info(f"Improved configuration loaded")
    
    try:
        # Load data
        logger.info("Loading data...")
        try:
            # Update this path to your actual data file
            df = pd.read_csv(r"F:\VesperSet\stock_data_analysis\data\raw\indexData\HS300_daily_kline.csv")
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            logger.warning(f"Failed to load data: {e}. Using synthetic data for testing.")
            
            # Generate improved synthetic data
            np.random.seed(42)
            n_days = 3000  # More data for better training
            dates = pd.date_range('2015-01-01', periods=n_days)
            
            # Generate realistic price data with trends and volatility
            base_return = 0.0002
            volatility = 0.02
            
            # Add some trending behavior
            trend = np.sin(np.arange(n_days) / 250) * 0.001  # Yearly cycle
            returns = np.random.normal(base_return, volatility, n_days) + trend
            
            # Add some regime changes
            regime_changes = [500, 1000, 1500, 2000, 2500]
            for change_point in regime_changes:
                if change_point < n_days:
                    returns[change_point:] += np.random.normal(0, 0.005, n_days - change_point)
            
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Generate OHLV data
            df = pd.DataFrame({
                'Date': dates,
                'Open': prices * np.random.uniform(0.995, 1.005, n_days),
                'High': prices * np.random.uniform(1.005, 1.03, n_days),
                'Low': prices * np.random.uniform(0.97, 0.995, n_days),
                'Close': prices,
                'Volume': np.random.randint(1000000, 20000000, n_days),
                'StockCode': ['TEST'] * n_days,
                'transaction_amount': prices * np.random.randint(1000000, 20000000, n_days),
                'Amplitude': np.abs(np.random.normal(0.02, 0.01, n_days)),
                'TurnoverRate': np.random.uniform(0.01, 0.1, n_days),
                'ChangePercentage': returns * 100,
                'ChangeAmount': np.diff(prices, prepend=prices[0])
            })
            
            logger.info(f"Improved synthetic data generated. Shape: {df.shape}")
        
        # Display data info
        logger.info("\nData Summary:")
        logger.info(f"Date range: {df.iloc[0, 0] if 'Date' in df.columns else 'No Date'} to {df.iloc[-1, 0] if 'Date' in df.columns else 'No Date'}")
        logger.info(f"Price range: {df['Close'].min():.2f} to {df['Close'].max():.2f}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        logger.info(f"Data types: {dict(df.dtypes)}")
        
        # Initialize and fit predictor
        predictor = ImprovedStockPredictor(config)
        results = predictor.fit(df)
        
        # Display final results
        logger.info("\n" + "="*70)
        logger.info("FINAL IMPROVED RESULTS")
        logger.info("="*70)
        logger.info(f"Mean CV Score: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        
        # Performance comparison with original
        original_score = 0.0784  # From your original run
        improvement = results['mean_score'] - original_score
        improvement_pct = (improvement / original_score) * 100 if original_score > 0 else float('inf')
        
        logger.info(f"Improvement over original: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        logger.info("Training completed successfully!")
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Improved pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    results = main()