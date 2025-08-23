# -*- coding: utf-8 -*-
# Optimized Stock CNN 2D with Enhanced Architecture and Training
"""
Key Optimizations:
1. **Memory Efficiency**: Optimized data structures and batch processing
2. **Model Architecture**: Enhanced CNN with attention, residual connections
3. **Training Strategy**: Improved loss functions, optimizers, and schedulers
4. **Feature Engineering**: Better feature selection and normalization
5. **Validation**: Robust time series cross-validation with proper metrics
6. **Performance**: GPU optimization, mixed precision training, data loading
"""

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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional imports with fallbacks
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# ENHANCED CONFIGURATION
# =========================

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Architecture type
    model_type: str = "enhanced_cnn"  # "enhanced_cnn", "inception", "resnet", "transformer"
    
    # CNN parameters
    base_channels: int = 16  # Reduced from 32 for lighter model
    channel_multiplier: float = 1.5  # Reduced from 2.0 for lighter model
    num_blocks: int = 2  # Reduced from 3 for lighter model
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5])  # Reduced kernel sizes
    
    # Attention parameters
    use_attention: bool = True
    attention_type: str = "se"  # Changed to lighter SE attention
    
    # Regularization
    dropout_rate: float = 0.4  # Increased from 0.3 for stronger regularization
    dropblock_rate: float = 0.15  # Increased from 0.1 for stronger regularization
    use_dropout2d: bool = True
    
    # Normalization
    norm_type: str = "group"  # "group", "instance", "layer" (avoid batch norm for small batches)
    
    # Activation
    activation: str = "swish"  # "relu", "swish", "gelu", "mish"

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    batch_size: int = 64
    epochs: int = 200
    early_stopping_patience: int = 25
    
    # Optimizer parameters
    optimizer_type: str = "adamw"  # "adamw", "adam", "sgd"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler parameters
    scheduler_type: str = "cosine_warm"  # "cosine_warm", "onecycle", "reduce_plateau" (start with cosine for stability)
    warmup_epochs: int = 5  # Reduced warmup to avoid overfitting
    max_lr_factor: float = 3.0  # Reduced peak LR to avoid oscillation
    
    # Loss function
    loss_type: str = "label_smoothing"  # "label_smoothing", "cross_entropy", "focal"
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05  # Reduced from 0.1 for better calibration
    
    # Mixed precision and optimization
    use_amp: bool = True
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Monitoring
    monitor_metric: str = "val_mcc"  # Changed from val_f1_macro to MCC for better stability
    monitor_mode: str = "max"

@dataclass
class DataConfig:
    """Data processing configuration"""
    # Data parameters
    window_size: int = 60
    stride: int = 1
    prediction_horizon: int = 5
    
    # Feature engineering
    feature_engineering_level: str = "expert"  # "basic", "advanced", "expert"
    max_features: int = 50
    feature_selection_method: str = "mutual_info"  # "mutual_info", "f_score", "variance"
    correlation_threshold: float = 0.95
    
    # Labeling strategy
    labeling_method: str = "volatility_adaptive"  # "binary", "ternary", "volatility_adaptive"
    binary_threshold: float = 0.005
    volatility_window: int = 252
    
    # Scaling
    scaler_type: str = "robust"  # "robust", "standard", "minmax"
    feature_wise_normalization: bool = True
    use_global_scaler: bool = True  # Set to False to use only window normalization
    
    # Time series validation
    validation_method: str = "walk_forward"  # "walk_forward", "expanding_window"
    n_splits: int = 5
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    embargo_period: int = 65  # window_size + prediction_horizon to avoid boundary leakage

@dataclass
class OptimizedConfig:
    """Complete optimized configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # System settings
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    
    # Logging and checkpoints
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    
    # Experiment tracking
    use_tensorboard: bool = True
    experiment_name: str = "stock_cnn_experiment"

# =========================
# ENHANCED DATA PROCESSING
# =========================

class AdvancedFeatureEngineer:
    """Advanced feature engineering with domain knowledge"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        logger.info("Starting advanced feature engineering...")
        df = df.copy()
        
        # Get column mapping
        self.column_mapping = self._detect_columns(df)
        
        # Basic technical indicators
        df = self._add_technical_indicators(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        # Advanced features based on level
        if self.config.feature_engineering_level in ["advanced", "expert"]:
            df = self._add_advanced_features(df)
            
        if self.config.feature_engineering_level == "expert":
            df = self._add_expert_features(df)
        
        # Feature interaction and polynomial features
        df = self._add_interaction_features(df)
        
        # Time-based features
        df = self._add_temporal_features(df)
        
        # Final NaN handling to ensure no None values remain
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        # Convert any remaining object columns to numeric
        for col in df.columns:
            if col != 'Date' and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df[col] = 0.0
        
        # Final NaN check
        df = df.fillna(0.0)
        
        # Ensure no datetime columns remain in the final dataframe
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
        if len(datetime_cols) > 0:
            logger.warning(f"Removing remaining datetime columns: {datetime_cols.tolist()}")
            df = df.drop(columns=datetime_cols)
        
        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        return df
    
    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect column names"""
        mapping = {}
        columns = df.columns.str.lower()
        
        # Price columns
        for col, pattern in [
            ('open', 'open'), ('high', 'high'), ('low', 'low'), 
            ('close', 'close'), ('volume', 'vol'), ('date', 'date')
        ]:
            matches = columns[columns.str.contains(pattern, na=False)]
            if len(matches) > 0:
                mapping[col] = df.columns[columns.get_loc(matches[0])]
        
        return mapping
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        close = df[self.column_mapping.get('close', 'Close')]
        high = df[self.column_mapping.get('high', 'High')]
        low = df[self.column_mapping.get('low', 'Low')]
        volume = df[self.column_mapping.get('volume', 'Volume')]
        
        # Price transforms
        df['log_close'] = np.log(close)
        df['log_returns'] = np.log(close / close.shift(1))
        df['log_volume'] = np.log(volume + 1)
        
        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = close.rolling(period).mean()
            df[f'ema_{period}'] = close.ewm(span=period).mean()
            df[f'price_sma_ratio_{period}'] = close / df[f'sma_{period}']
            
        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = close.pct_change(period)
            df[f'momentum_{period}'] = close - close.shift(period)
            
        # Volatility indicators
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['log_returns'].rolling(period).std()
            df[f'parkinson_{period}'] = np.sqrt(
                np.log(high / low) ** 2 / (4 * np.log(2))
            ).rolling(period).mean()
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_mean = close.rolling(period).mean()
            bb_std = close.rolling(period).std()
            df[f'bb_upper_{period}'] = bb_mean + 2 * bb_std
            df[f'bb_lower_{period}'] = bb_mean - 2 * bb_std
            df[f'bb_position_{period}'] = (close - bb_mean) / (2 * bb_std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_mean
        
        # RSI
        for period in [14, 30]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Handle NaN values that might have been created
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        close = df[self.column_mapping.get('close', 'Close')]
        high = df[self.column_mapping.get('high', 'High')]
        low = df[self.column_mapping.get('low', 'Low')]
        volume = df[self.column_mapping.get('volume', 'Volume')]
        
        # Spread proxies
        df['hl_spread'] = (high - low) / close
        df['oc_spread'] = abs(close - df[self.column_mapping.get('open', 'Open')]) / close
        
        # Volume-price relationships
        df['vwap'] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        df['price_vwap_ratio'] = close / df['vwap']
        
        # Order flow proxies
        typical_price = (high + low + close) / 3
        df['mfi'] = self._money_flow_index(typical_price, volume)
        df['obv'] = self._on_balance_volume(close, volume)
        
        # Liquidity proxies
        df['amihud_illiq'] = abs(df['log_returns']) / (volume * close + 1e-8)
        df['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        
        # Handle NaN values that might have been created
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        return df
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced quantitative features"""
        close = df[self.column_mapping.get('close', 'Close')]
        
        # Statistical features
        for window in [10, 20]:
            returns = df['log_returns']
            df[f'skewness_{window}'] = returns.rolling(window).skew()
            df[f'kurtosis_{window}'] = returns.rolling(window).kurt()
            df[f'jarque_bera_{window}'] = (df[f'skewness_{window}']**2 + df[f'kurtosis_{window}']**2/4) / 6
        
        # Regime detection features
        df['price_acceleration'] = df['log_returns'].diff()
        df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(60).quantile(0.8)).astype(int)
        
        # Fractal features
        df['hurst_exponent'] = self._rolling_hurst(df['log_returns'], window=60)
        df['fractal_dimension'] = 2 - df['hurst_exponent']
        
        # Information theory features
        df['entropy_20'] = -df['log_returns'].rolling(20).apply(
            lambda x: np.sum(np.histogram(x, bins=10, density=True)[0] * 
                           np.log(np.histogram(x, bins=10, density=True)[0] + 1e-8))
        )
        
        # Handle NaN values that might have been created
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        return df
    
    def _add_expert_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add expert-level quantitative features"""
        close = df[self.column_mapping.get('close', 'Close')]
        
        # Options-inspired features (using realized volatility)
        df['iv_rank'] = df['volatility_20'].rolling(252).rank(pct=True)
        df['vol_skew'] = df['volatility_5'] / df['volatility_20'] - 1
        
        # Correlation features
        returns = df['log_returns']
        volume = df['log_volume']
        df['price_volume_corr'] = returns.rolling(60).corr(volume)
        
        # Regime switching indicators
        df['trend_strength'] = abs(df['sma_20'] - df['sma_20'].shift(20)) / df['sma_20'].shift(20)
        df['mean_reversion_speed'] = -np.log(abs(df['price_sma_ratio_20'] - 1) + 1e-8)
        
        # Risk-adjusted features
        df['sharpe_20'] = df['log_returns'].rolling(20).mean() / (df['volatility_20'] + 1e-8)
        df['sortino_20'] = df['log_returns'].rolling(20).mean() / (
            df['log_returns'].rolling(20).apply(lambda x: x[x < 0].std()) + 1e-8
        )
        
        # Jump detection
        df['jump_indicator'] = (abs(df['log_returns']) > 3 * df['volatility_20']).astype(int)
        df['jump_intensity'] = df['jump_indicator'].rolling(20).sum()
        
        # Handle NaN values that might have been created
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions"""
        # Key interaction features
        df['momentum_vol_interaction'] = df['roc_10'] * df['volatility_10']
        df['rsi_bb_interaction'] = df['rsi_14'] * df['bb_position_20']
        df['ma_cross_strength'] = (df['sma_5'] - df['sma_20']) / df['sma_50']
        
        # Volume-price interactions
        df['vol_price_momentum'] = df['volume_ma_ratio'] * df['roc_5']
        df['vol_volatility_regime'] = df['volume_ma_ratio'] * df['volatility_regime']
        
        # Handle NaN values that might have been created
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['quarter'] = df['Date'].dt.quarter
            df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
            df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
            
            # Remove the original Date column to avoid passing datetime to numeric operations
            df = df.drop(columns=['Date'])
        
        # Handle NaN values that might have been created
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        return df
    
    def _money_flow_index(self, typical_price: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window).sum()
        return 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))
    
    def _on_balance_volume(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume"""
        obv = np.where(close > close.shift(1), volume, 
                      np.where(close < close.shift(1), -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()
    
    def _rolling_hurst(self, series: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling Hurst exponent"""
        def hurst_exponent(ts):
            try:
                ts = np.array(ts)
                n = len(ts)
                if n < 20:
                    return 0.5
                
                # Calculate R/S statistic
                lags = range(2, min(n//2, 20))
                rs_values = []
                
                for lag in lags:
                    # Divide into segments
                    segments = n // lag
                    rs_segment = []
                    
                    for i in range(segments):
                        start_idx = i * lag
                        end_idx = start_idx + lag
                        segment = ts[start_idx:end_idx]
                        
                        mean_segment = np.mean(segment)
                        deviations = segment - mean_segment
                        cumulative_deviations = np.cumsum(deviations)
                        
                        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                        S = np.std(segment)
                        
                        if S > 0:
                            rs_segment.append(R / S)
                    
                    if rs_segment:
                        rs_values.append(np.mean(rs_segment))
                
                if len(rs_values) < 5:
                    return 0.5
                
                # Linear regression to find Hurst exponent
                log_lags = np.log(lags[:len(rs_values)])
                log_rs = np.log(rs_values)
                
                coeffs = np.polyfit(log_lags, log_rs, 1)
                return max(0.01, min(0.99, coeffs[0]))
            except:
                return 0.5
        
        return series.rolling(window).apply(hurst_exponent, raw=False)

class OptimizedLabelGenerator:
    """Optimized label generation with multiple strategies"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate labels based on configuration"""
        df = df.copy()
        
        if self.config.labeling_method == "binary":
            return self._binary_labels(df)
        elif self.config.labeling_method == "ternary":
            return self._ternary_labels(df)
        elif self.config.labeling_method == "volatility_adaptive":
            return self._volatility_adaptive_labels(df)
        else:
            raise ValueError(f"Unknown labeling method: {self.config.labeling_method}")
    
    def _binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary classification labels"""
        close_col = self._get_close_column(df)
        
        # Calculate forward returns
        forward_returns = df[close_col].shift(-self.config.prediction_horizon) / df[close_col] - 1.0
        
        # Create binary labels (0: down, 1: up) - use float to handle NaN safely
        labels = pd.Series(index=df.index, dtype=float)
        labels[forward_returns > self.config.binary_threshold] = 1.0
        labels[forward_returns <= -self.config.binary_threshold] = 0.0
        
        # Remove neutral samples first
        valid_mask = ~labels.isna()
        df = df[valid_mask].copy()
        labels = labels[valid_mask]
        
        # Convert to int after filtering out NaN values
        labels = labels.astype(int)
        
        df['target'] = labels
        logger.info(f"Binary labels - Up: {(labels == 1).sum()}, Down: {(labels == 0).sum()}")
        return df
    
    def _ternary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ternary classification labels"""
        close_col = self._get_close_column(df)
        forward_returns = df[close_col].shift(-self.config.prediction_horizon) / df[close_col] - 1.0
        
        # Use quantile-based thresholds
        upper_threshold = forward_returns.rolling(self.config.volatility_window).quantile(0.66)
        lower_threshold = forward_returns.rolling(self.config.volatility_window).quantile(0.33)
        
        # Create labels with float dtype to handle NaN safely
        labels = pd.Series(1.0, index=df.index, dtype=float)  # Default: neutral
        labels[forward_returns > upper_threshold] = 2.0  # Up
        labels[forward_returns < lower_threshold] = 0.0  # Down
        
        # Remove samples with NaN values first
        valid_mask = ~labels.isna()
        df = df[valid_mask].copy()
        labels = labels[valid_mask]
        
        # Convert to int after filtering out NaN values
        labels = labels.astype(int)
        
        df['target'] = labels
        logger.info(f"Ternary labels - Down: {(labels == 0).sum()}, "
                   f"Neutral: {(labels == 1).sum()}, Up: {(labels == 2).sum()}")
        return df
    
    def _volatility_adaptive_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-adaptive labels"""
        close_col = self._get_close_column(df)
        
        # Calculate returns and volatility
        returns = df[close_col].pct_change()
        volatility = returns.rolling(self.config.volatility_window).std()
        
        # Calculate forward returns
        forward_returns = df[close_col].shift(-self.config.prediction_horizon) / df[close_col] - 1.0
        
        # Adaptive thresholds based on current volatility
        current_vol = volatility.shift(1)  # Use past volatility to avoid lookahead bias
        upper_threshold = current_vol * 1.0  # 1 standard deviation
        lower_threshold = -current_vol * 1.0
        
        # Use np.select for efficient label generation
        conditions = [
            forward_returns > upper_threshold,  # UP condition
            forward_returns < lower_threshold   # DOWN condition
        ]
        choices = [2, 0]  # [UP_label, DOWN_label]
        labels = np.select(conditions, choices, default=1)
        
        df['target'] = labels
        logger.info(f"Volatility-adaptive labels - Down: {(labels == 0).sum()}, "
                   f"Neutral: {(labels == 1).sum()}, Up: {(labels == 2).sum()}")
        return df
    
    def _get_close_column(self, df: pd.DataFrame) -> str:
        """Get close column name"""
        for col in ['Close', 'close', 'CLOSE']:
            if col in df.columns:
                return col
        raise ValueError("No close column found")

class OptimizedDataProcessor:
    """Optimized data processing pipeline"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data - only feature engineering and labeling, no scaling/selection"""
        # Feature engineering
        feature_engineer = AdvancedFeatureEngineer(self.config)
        df = feature_engineer.engineer_features(df)
        
        # Label generation
        label_generator = OptimizedLabelGenerator(self.config)
        df = label_generator.generate_labels(df)
        
        # Note: Feature selection and scaling will be done per-fold to avoid data leakage
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted processors"""
        # Feature engineering (without fitting)
        feature_engineer = AdvancedFeatureEngineer(self.config)
        df = feature_engineer.engineer_features(df)
        
        # Use previously selected features and fitted scaler
        feature_cols = [col for col in df.columns if col != 'target' and col in self.selected_features]
        X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create output dataframe
        result_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
        if 'target' in df.columns:
            result_df['target'] = df['target']
        
        return result_df
    
    def _select_and_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features and fit scaler"""
        # Separate features and targets
        feature_cols = [col for col in df.columns if col not in ['target', 'Date', 'date']]
        X = df[feature_cols].values
        y = df['target'].values if 'target' in df.columns else None
        
        # Additional safety check for None values
        if X is None or y is None:
            raise ValueError("Data contains None values after feature engineering")
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature selection
        if len(feature_cols) > self.config.max_features:
            X, selected_indices = self._select_features(X, y, feature_cols)
            self.selected_features = [feature_cols[i] for i in selected_indices]
        else:
            self.selected_features = feature_cols
        
        # Initialize scaler
        if self.config.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.config.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")
        
        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Create output dataframe
        result_df = pd.DataFrame(X_scaled, columns=self.selected_features, index=df.index)
        if y is not None:
            result_df['target'] = y
        
        logger.info(f"Selected {len(self.selected_features)} features after selection and scaling")
        return result_df
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[int]]:
        """Feature selection with correlation filtering"""
        # Remove highly correlated features
        X_filtered, keep_indices = self._remove_correlated_features(X, list(range(X.shape[1])))
        
        # Apply feature selection
        if self.config.feature_selection_method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=min(self.config.max_features, X_filtered.shape[1]))
        elif self.config.feature_selection_method == "f_score":
            selector = SelectKBest(score_func=f_classif, k=min(self.config.max_features, X_filtered.shape[1]))
        else:
            # Just use correlation filtering
            return X_filtered, keep_indices
        
        X_selected = selector.fit_transform(X_filtered, y)
        selected_mask = selector.get_support()
        
        # Map back to original indices
        final_indices = [keep_indices[i] for i in range(len(keep_indices)) if selected_mask[i]]
        
        logger.info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
        return X_selected, final_indices
    
    def _remove_correlated_features(self, X: np.ndarray, feature_indices: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Remove highly correlated features using only training data"""
        if X.shape[1] <= 1:
            return X, list(range(X.shape[1]))
        
        # Calculate correlation matrix with handling for constant features
        try:
            # Remove constant features first (std = 0)
            feature_stds = np.std(X, axis=0)
            valid_features = feature_stds > 1e-8
            
            if not np.any(valid_features):
                # All features are constant, keep the first one
                return X[:, :1], [0]
            
            X_filtered = X[:, valid_features]
            valid_indices = np.where(valid_features)[0]
            
            if X_filtered.shape[1] <= 1:
                return X_filtered, valid_indices.tolist()
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X_filtered.T)
            
            # Handle NaN values (can occur with very small variations)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr_matrix, 0)
            
        except Exception as e:
            logger.warning(f"Error in correlation calculation: {e}. Using all features.")
            return X, list(range(X.shape[1]))
        
        # Find features to remove from filtered set
        to_remove = set()
        for i in range(corr_matrix.shape[0]):
            if i in to_remove:
                continue
            for j in range(i + 1, corr_matrix.shape[1]):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > self.config.correlation_threshold:
                    # Keep the feature with lower index (usually simpler)
                    to_remove.add(max(i, j))
        
        # Keep features that are not highly correlated from the filtered set
        keep_indices_filtered = np.array([i for i in range(X_filtered.shape[1]) if i not in to_remove])
        final_indices = valid_indices[keep_indices_filtered]
        
        logger.info(f"Removed {len(to_remove)} highly correlated features (|ρ| > {self.config.correlation_threshold})")
        return X[:, final_indices], final_indices.tolist()

# =========================
# ENHANCED MODEL ARCHITECTURES
# =========================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ECABlock(nn.Module):
    """Efficient Channel Attention block"""
    def __init__(self, channels: int, gamma: int = 2, beta: int = 1):
        super().__init__()
        t = int(abs(np.log2(channels) + beta) / gamma)
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DropBlock2D(nn.Module):
    """DropBlock regularization for 2D features"""
    def __init__(self, drop_rate: float = 0.1, block_size: int = 7):
        super().__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate == 0:
            return x
        
        gamma = self.drop_rate / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        
        # Enlarge mask
        mask = F.max_pool2d(mask[:, None, :, :], 
                           kernel_size=self.block_size, 
                           stride=1, 
                           padding=self.block_size // 2)
        mask = mask.squeeze(1)
        
        # Normalize
        mask = 1 - mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            normalize_factor = mask.numel() / mask_sum
        else:
            normalize_factor = 1.0  # Fallback if all elements are masked
        
        return x * mask.unsqueeze(1) * normalize_factor

class ResidualBlock(nn.Module):
    """Residual block with attention and normalization"""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 config: ModelConfig,
                 stride: int = 1):
        super().__init__()
        
        # Debug: Log block parameters
        logger.info(f"Initializing ResidualBlock: in_channels={in_channels}, out_channels={out_channels}, "
                   f"stride={stride}, norm_type={config.norm_type}, activation={config.activation}")
        
        # Validate parameters
        if in_channels is None or in_channels <= 0:
            raise ValueError(f"Invalid in_channels: {in_channels}")
        if out_channels is None or out_channels <= 0:
            raise ValueError(f"Invalid out_channels: {out_channels}")
        if config.norm_type is None:
            raise ValueError(f"Invalid norm_type: {config.norm_type}")
        if config.activation is None:
            raise ValueError(f"Invalid activation: {config.activation}")
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.norm1 = self._get_norm_layer(out_channels, config.norm_type)
        self.activation1 = self._get_activation(config.activation)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.norm2 = self._get_norm_layer(out_channels, config.norm_type)
        
        # Attention
        if config.use_attention:
            if config.attention_type == "se":
                self.attention = SEBlock(out_channels)
            elif config.attention_type == "eca":
                self.attention = ECABlock(out_channels)
            elif config.attention_type == "cbam":
                self.attention = CBAMBlock(out_channels)
            else:
                self.attention = nn.Identity()
        else:
            self.attention = nn.Identity()
        
        # Shortcut connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                self._get_norm_layer(out_channels, config.norm_type)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Regularization
        self.dropout = nn.Dropout2d(config.dropout_rate) if config.use_dropout2d else nn.Identity()
        self.dropblock = DropBlock2D(config.dropblock_rate) if config.dropblock_rate > 0 else nn.Identity()
        
        self.final_activation = self._get_activation(config.activation)
        
        logger.info(f"ResidualBlock initialization completed successfully")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.attention(out)
        out = self.dropblock(out)
        
        # Shortcut connection
        residual = self.shortcut(residual)
        out += residual
        
        out = self.final_activation(out)
        return out
    
    def _get_norm_layer(self, channels: int, norm_type: str) -> nn.Module:
        logger.info(f"Creating norm layer: channels={channels}, norm_type={norm_type}")
        
        if norm_type is None:
            raise ValueError(f"norm_type is None")
        if channels is None or channels <= 0:
            raise ValueError(f"Invalid channels: {channels}")
        
        if norm_type == "batch":
            return nn.BatchNorm2d(channels)
        elif norm_type == "instance":
            return nn.InstanceNorm2d(channels)
        elif norm_type == "layer":
            return nn.GroupNorm(1, channels)
        elif norm_type == "group":
            # More robust GroupNorm logic
            if channels % 32 == 0:
                return nn.GroupNorm(32, channels)
            elif channels % 16 == 0:
                return nn.GroupNorm(16, channels)
            elif channels % 8 == 0:
                return nn.GroupNorm(8, channels)
            else:
                # Fallback to LayerNorm if not easily divisible
                return nn.GroupNorm(1, channels)
        else:
            logger.warning(f"Unknown norm_type: {norm_type}, using Identity")
            return nn.Identity()
    
    def _get_activation(self, activation: str) -> nn.Module:
        logger.info(f"Creating activation layer: activation={activation}")
        
        if activation is None:
            raise ValueError(f"activation is None")
        
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "swish":
            return nn.SiLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "mish":
            return nn.Mish(inplace=True)
        else:
            logger.warning(f"Unknown activation: {activation}, using ReLU")
            return nn.ReLU(inplace=True)

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction module"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Ensure minimum channel count for multi-scale processing
        if out_channels < 4:
            # For very small channel counts, use a simpler approach
            self.convs = nn.ModuleList([
                nn.Conv2d(in_channels, 1, kernel_size=(k, 1), padding=(k // 2, 0))
                for k in [1, 3, 5, 7]
            ])
            # Output will be 4 channels, then project to desired output
            self.output_proj = nn.Conv2d(4, out_channels, 1)
        else:
            # Dynamically allocate channels for larger counts
            c1 = max(1, out_channels // 4)  # Ensure at least 1 channel
            c2 = max(1, out_channels // 4)
            c3 = max(1, out_channels // 4)
            c4 = max(1, out_channels - (c1 + c2 + c3))  # Ensure remainder is at least 1
            
            channels = [c1, c2, c3, c4]
            
            self.convs = nn.ModuleList([
                nn.Conv2d(in_channels, ch, kernel_size=(k, 1), padding=(k // 2, 0))
                for k, ch in zip([1, 3, 5, 7], channels)
            ])
            self.output_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [conv(x) for conv in self.convs]
        x = torch.cat(out, dim=1)
        if self.output_proj:
            x = self.output_proj(x)
        return x

class OptimizedStockCNN(nn.Module):
    """Optimized CNN architecture for stock prediction"""
    def __init__(self, num_classes: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Debug: Log configuration values
        logger.info(f"Initializing OptimizedStockCNN with: num_classes={num_classes}, "
                   f"base_channels={config.base_channels}, "
                   f"channel_multiplier={config.channel_multiplier}, "
                   f"num_blocks={config.num_blocks}")
        
        # Validate configuration
        if config.base_channels is None or config.base_channels <= 0:
            raise ValueError(f"Invalid base_channels: {config.base_channels}")
        if config.channel_multiplier is None or config.channel_multiplier <= 0:
            raise ValueError(f"Invalid channel_multiplier: {config.channel_multiplier}")
        if config.num_blocks is None or config.num_blocks <= 0:
            raise ValueError(f"Invalid num_blocks: {config.num_blocks}")
        
        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, config.base_channels, kernel_size=3, padding=1),
            self._get_norm_layer(config.base_channels, config.norm_type),
            self._get_activation(config.activation)
        )
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(config.base_channels, config.base_channels)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        in_channels = config.base_channels
        
        for i in range(config.num_blocks):
            out_channels = int(config.base_channels * (config.channel_multiplier ** i))
            stride = 2 if i > 0 else 1  # Downsample after first block
            
            logger.info(f"Creating ResidualBlock {i}: in_channels={in_channels}, out_channels={out_channels}")
            
            self.blocks.append(
                ResidualBlock(in_channels, out_channels, config, stride)
            )
            in_channels = out_channels
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(in_channels, in_channels // 2),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(in_channels // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        logger.info("OptimizedStockCNN initialization completed successfully")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input processing
        x = self.input_conv(x)
        
        # Multi-scale features
        x = self.multi_scale(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x
    def _get_norm_layer(self, channels: int, norm_type: str) -> nn.Module: 
        if norm_type == "batch": 
            return nn.BatchNorm2d(channels) 
        elif norm_type == "instance": 
            return nn.InstanceNorm2d(channels) 
        elif norm_type == "layer": 
            return nn.GroupNorm(1, channels) 
        elif norm_type == "group": 
            return nn.GroupNorm(min(32, channels), channels) 
        else: 
            return nn.Identity() 
    def _get_activation(self, activation: str) -> nn.Module: 
        if activation == "relu": 
            return nn.ReLU(inplace=True) 
        elif activation == "swish": 
            return nn.SiLU(inplace=True) 
        elif activation == "gelu": 
            return nn.GELU() 
        elif activation == "mish": 
            return nn.Mish(inplace=True) 
        else: 
            return nn.ReLU(inplace=True)
    

    
    def _get_activation(self, activation: str) -> nn.Module:
        logger.info(f"Creating activation layer: activation={activation}")
        
        if activation is None:
            raise ValueError(f"activation is None")
        
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "swish":
            return nn.SiLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "mish":
            return nn.Mish(inplace=True)
        else:
            logger.warning(f"Unknown activation: {activation}, using ReLU")
            return nn.ReLU(inplace=True)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# =========================
# ENHANCED LOSS FUNCTIONS
# =========================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



# =========================
# ENHANCED DATASET
# =========================

def safe_isnan(arr):
    """Safely check for NaN values in any numpy array"""
    try:
        if arr.dtype.kind in 'fc':  # float or complex
            return np.isnan(arr)
        else:
            # For non-float types, check if any values are None or invalid
            return np.array([False] * arr.size).reshape(arr.shape)
    except:
        return np.array([False] * arr.size).reshape(arr.shape)

class OptimizedStockDataset(Dataset):
    """Optimized dataset with efficient memory usage"""
    
    def __init__(self, 
                 data: np.ndarray, 
                 labels: np.ndarray, 
                 window_size: int,
                 stride: int = 1,
                 transform=None):
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        # Create sequences efficiently
        self.sequences, self.targets = self._create_sequences(data, labels)
        
        # Memory optimization: use float32
        self.sequences = self.sequences.astype(np.float32)
        self.targets = self.targets.astype(np.int64)
        
        logger.info(f"Dataset created: {len(self.sequences)} samples, "
                   f"sequence shape: {self.sequences.shape[1:]}")
    
    def _create_sequences(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences using stride tricks for memory efficiency"""
        # Safety check for None values
        if data is None or labels is None:
            raise ValueError("Data or labels contain None values")
        
        n_samples, n_features = data.shape
        n_sequences = (n_samples - self.window_size) // self.stride + 1
        
        # Use stride tricks for efficient sequence creation
        shape = (n_sequences, self.window_size, n_features)
        strides = (data.strides[0] * self.stride, data.strides[0], data.strides[1])
        sequences = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        
        # Get corresponding labels
        label_indices = np.arange(self.window_size - 1, n_samples, self.stride)[:n_sequences]
        targets = labels[label_indices]
        
        # Ensure proper data types
        sequences = sequences.astype(np.float32)
        targets = targets.astype(np.int64)
        
        # Add channel dimension for CNN
        sequences = sequences[:, None, :, :]
        
        # Final safety check for NaN/None values
        if np.any(safe_isnan(sequences)) or np.any(safe_isnan(targets)):
            sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
            targets = np.nan_to_num(targets, nan=0, posinf=0, neginf=0)
        
        return sequences.copy(), targets.copy()
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # Safety check for None values
        if sequence is None or target is None:
            raise ValueError(f"Sequence or target at index {idx} is None")
        
        # Apply transforms if provided
        if self.transform:
            sequence = self.transform(sequence)
        
        # Window-wise normalization for better training stability
        #sequence = self._normalize_window(sequence)
        
        # Final safety check before returning
        if np.any(safe_isnan(sequence)):
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.from_numpy(sequence), torch.tensor(target)
    
    def _normalize_window(self, sequence: np.ndarray) -> np.ndarray:
        """Apply window-wise normalization"""
        # Safety check for None values
        if sequence is None:
            raise ValueError("Sequence is None in _normalize_window")
        
        # sequence shape: (1, window_size, n_features)
        window_data = sequence[0]  # Remove channel dimension
        
        # Safety check for NaN values
        if np.any(safe_isnan(window_data)):
            window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Z-score normalization within window
        window_mean = np.mean(window_data, axis=0, keepdims=True)
        window_std = np.std(window_data, axis=0, keepdims=True)
        window_std = np.where(window_std == 0, 1e-8, window_std)
        
        normalized = (window_data - window_mean) / window_std
        
        # Clip extreme values
        normalized = np.clip(normalized, -5, 5)
        
        # Final safety check
        if np.any(safe_isnan(normalized)):
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized[None, :, :]  # Add channel dimension back

# =========================
# ENHANCED TRAINING PIPELINE
# =========================

class OptimizedTrainer:
    """Optimized trainer with comprehensive features"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        
        # Windows compatibility: disable num_workers to avoid fork issues
        if platform.system().lower().startswith('win'):
            self.config.num_workers = 0
            logger.info("Windows detected: setting num_workers=0 for compatibility")
        
        self.device = self._get_device()
        
        # Initialize logging
        self.writer = None
        if config.use_tensorboard:
            log_dir = Path(config.log_dir) / config.experiment_name
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Temperature scaling for calibration
        self.temperature = 1.0
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
    
    def fit(self, train_data: np.ndarray, train_labels: np.ndarray,
            val_data: np.ndarray, val_labels: np.ndarray) -> Dict[str, Any]:
        """Main training loop"""
        
        # Validate input data
        if train_data is None or train_labels is None or val_data is None or val_labels is None:
            raise ValueError("Input data or labels contain None values")
        
        if len(train_data) == 0 or len(val_data) == 0:
            raise ValueError("Empty training or validation data")
        
        # Create datasets
        train_dataset = OptimizedStockDataset(
            train_data, train_labels, 
            self.config.data.window_size, 
            self.config.data.stride
        )
        
        val_dataset = OptimizedStockDataset(
            val_data, val_labels,
            self.config.data.window_size,
            stride=1  # No stride for validation
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,  # Same batch size as training for consistent normalization
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        # Initialize model and training components
        logger.info(f"Setting up training with {len(train_loader)} steps per epoch")
        self._setup_training(train_labels, len(train_loader))
        
        logger.info(f"Starting training for {self.config.training.epochs} epochs")
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Training loop
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self._train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.config.training.monitor_metric])
                elif not isinstance(self.scheduler, OneCycleLR):
                    # OneCycleLR is stepped per batch, others per epoch
                    self.scheduler.step()
            else:
                logger.warning("No scheduler available for learning rate updates")
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Check for improvement and early stopping
            current_score = val_metrics[self.config.training.monitor_metric]
            if self._is_improvement(current_score):
                self.best_score = current_score
                self.patience_counter = 0
                self._save_checkpoint('best_model.pth')
                logger.info(f"New best score: {current_score:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Memory cleanup
            if epoch % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        # Fit temperature scaling on validation set for better calibration
        if hasattr(self, 'val_logits') and hasattr(self, 'val_labels'):
            self._fit_temperature_scaling()
        
        return {
            'best_score': self.best_score,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'epochs_trained': self.current_epoch + 1,
            'temperature': self.temperature
        }
    
    def _setup_training(self, train_labels: np.ndarray, steps_per_epoch: int):
        """Setup model, optimizer, scheduler, and loss function"""
        # Validate steps_per_epoch
        if steps_per_epoch <= 0:
            raise ValueError(f"Invalid steps_per_epoch: {steps_per_epoch}. Must be positive.")
        
        num_classes = len(np.unique(train_labels))
        
        # Debug: Log model configuration
        logger.info(f"Model config: base_channels={self.config.model.base_channels}, "
                   f"channel_multiplier={self.config.model.channel_multiplier}, "
                   f"num_blocks={self.config.model.num_blocks}, "
                   f"dropout_rate={self.config.model.dropout_rate}, "
                   f"attention_type={self.config.model.attention_type}, "
                   f"norm_type={self.config.model.norm_type}, "
                   f"activation={self.config.model.activation}")
        
        # Validate model configuration
        if self.config.model.base_channels is None or self.config.model.base_channels <= 0:
            raise ValueError(f"Invalid base_channels: {self.config.model.base_channels}")
        if self.config.model.channel_multiplier is None or self.config.model.channel_multiplier <= 0:
            raise ValueError(f"Invalid channel_multiplier: {self.config.model.channel_multiplier}")
        if self.config.model.num_blocks is None or self.config.model.num_blocks <= 0:
            raise ValueError(f"Invalid num_blocks: {self.config.model.num_blocks}")
        if self.config.model.dropout_rate is None or self.config.model.dropout_rate < 0:
            raise ValueError(f"Invalid dropout_rate: {self.config.model.dropout_rate}")
        if self.config.model.attention_type is None:
            raise ValueError(f"Invalid attention_type: {self.config.model.attention_type}")
        if self.config.model.norm_type is None:
            raise ValueError(f"Invalid norm_type: {self.config.model.norm_type}")
        if self.config.model.activation is None:
            raise ValueError(f"Invalid activation: {self.config.model.activation}")
        
        # Additional validation for specific values
        if self.config.model.base_channels < 1:
            raise ValueError(f"base_channels must be at least 1, got {self.config.model.base_channels}")
        if self.config.model.channel_multiplier < 1.0:
            raise ValueError(f"channel_multiplier must be at least 1.0, got {self.config.model.channel_multiplier}")
        if self.config.model.num_blocks < 1:
            raise ValueError(f"num_blocks must be at least 1, got {self.config.model.num_blocks}")
        if self.config.model.dropout_rate > 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self.config.model.dropout_rate}")
        
        # Initialize model
        try:
            self.model = OptimizedStockCNN(num_classes, self.config.model)
            self.model.to(self.device)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error(f"Model config: {self.config.model}")
            raise
        
        # Initialize optimizer
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
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas
            )
        elif self.config.training.optimizer_type == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )

        else:
            raise ValueError(f"Unknown optimizer type: {self.config.training.optimizer_type}")

        
        # Initialize scheduler
        if self.config.training.scheduler_type == "onecycle":
            logger.info(f"Initializing OneCycleLR scheduler with steps_per_epoch: {steps_per_epoch}")
            logger.info(f"OneCycleLR parameters: max_lr={self.config.training.learning_rate * self.config.training.max_lr_factor:.6f}, epochs={self.config.training.epochs}, pct_start={self.config.training.warmup_epochs / self.config.training.epochs:.3f}")
            
            # Validate parameters before creating scheduler
            if steps_per_epoch <= 0:
                raise ValueError(f"Invalid steps_per_epoch for OneCycleLR: {steps_per_epoch}")
            if self.config.training.epochs <= 0:
                raise ValueError(f"Invalid epochs for OneCycleLR: {self.config.training.epochs}")
            if self.config.training.warmup_epochs < 0 or self.config.training.warmup_epochs >= self.config.training.epochs:
                raise ValueError(f"Invalid warmup_epochs for OneCycleLR: {self.config.training.warmup_epochs}")
            
            # Additional safety checks for OneCycleLR parameters
            if self.config.training.learning_rate is None or self.config.training.max_lr_factor is None:
                logger.warning("OneCycleLR parameters contain None values, falling back to cosine_warm scheduler")
                self.config.training.scheduler_type = "cosine_warm"
            
            try:
                logger.info(f"Initializing OneCycleLR scheduler with steps_per_epoch: {steps_per_epoch}")
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=self.config.training.learning_rate * self.config.training.max_lr_factor,
                    epochs=self.config.training.epochs,
                    steps_per_epoch=steps_per_epoch,  # USE THE CORRECT, PASSED-IN VALUE
                    pct_start=self.config.training.warmup_epochs / self.config.training.epochs
                )
                logger.info("OneCycleLR scheduler initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OneCycleLR scheduler: {e}")
                logger.error(f"Parameters: max_lr={self.config.training.learning_rate * self.config.training.max_lr_factor:.6f}, epochs={self.config.training.epochs}, steps_per_epoch={steps_per_epoch}, pct_start={self.config.training.warmup_epochs / self.config.training.epochs:.3f}")
                logger.warning("Falling back to cosine_warm scheduler due to OneCycleLR initialization failure")
                self.config.training.scheduler_type = "cosine_warm"
        
        # Initialize cosine_warm scheduler (either as primary choice or fallback)
        if self.config.training.scheduler_type == "cosine_warm":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, self.config.training.epochs // 4),  # Ensure T_0 is at least 1
                T_mult=2
            )
            logger.info("CosineAnnealingWarmRestarts scheduler initialized successfully")
        
        # Initialize loss function
        if self.config.training.loss_type == "focal":
            # Calculate per-class alpha based on class distribution
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / (class_counts + 1e-8)  # Avoid division by zero
            class_weights = class_weights / class_weights.sum()  # Normalize
            alpha_tensor = torch.FloatTensor(class_weights).to(self.device)
            
            self.criterion = FocalLoss(
                alpha=alpha_tensor,  # Use per-class alpha
                gamma=self.config.training.focal_gamma
            )
        elif self.config.training.loss_type == "label_smoothing":
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=self.config.training.label_smoothing
            )
        else:
            # Use class weights for standard cross entropy
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(train_labels),
                y=train_labels
            )
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Validate scheduler initialization
        if self.scheduler is not None:
            logger.info(f"Scheduler initialized: {type(self.scheduler).__name__}")
            if isinstance(self.scheduler, OneCycleLR):
                try:
                    logger.info(f"OneCycleLR total steps: {self.scheduler.total_steps}")
                except Exception as e:
                    logger.warning(f"Could not access OneCycleLR total_steps: {e}")
        else:
            logger.warning("No scheduler initialized")
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.training.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config.training.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            preds = output.argmax(dim=1).cpu().numpy()
            targets = target.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Step scheduler for OneCycleLR (step per batch)
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                try:
                    self.scheduler.step()
                except Exception as e:
                    logger.error(f"Error stepping OneCycleLR scheduler at batch {batch_idx}: {e}")
                    logger.error(f"Scheduler state: {self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else 'No state_dict'}")
                    # Don't raise here, just log the error and continue
                    logger.warning("Continuing training despite scheduler stepping error")
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Ensure data is properly formatted for sklearn functions
        all_targets_array = np.array(all_targets, dtype=int)
        all_preds_array = np.array(all_preds, dtype=int)
        
        f1_macro = f1_score(all_targets_array, all_preds_array, average='macro')
        
        metrics = {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_f1_macro': f1_macro
        }
        
        self.train_metrics.append(metrics)
        return metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one epoch with rejection/neutral zone logic"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        all_probs = []
        all_preds_with_rejection = []  # Predictions with rejection logic
        all_logits = []  # Store logits for temperature scaling
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if self.config.training.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Update metrics
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                # Store logits for temperature scaling
                all_logits.extend(output.cpu().numpy().tolist())
                
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                # Apply rejection logic for ternary classification
                if probs.shape[1] == 3:  # Ternary classification
                    max_probs, second_probs = torch.topk(probs, 2, dim=1)
                    confidence_gap = max_probs[:, 0] - second_probs[:, 0]
                    
                    # Reject uncertain predictions (set to neutral/1)
                    rejection_mask = (max_probs[:, 0] < 0.5) | (confidence_gap < 0.1)
                    preds_with_rejection = preds.clone()
                    preds_with_rejection[rejection_mask] = 1  # Neutral class
                    
                    all_preds_with_rejection.extend(preds_with_rejection.cpu().numpy().tolist())
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(target.cpu().numpy().tolist())
        
        # Calculate comprehensive metrics
        avg_loss = total_loss / total_samples
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Ensure data is properly formatted for sklearn functions
        all_targets_array = np.array(all_targets, dtype=int)
        all_preds_array = np.array(all_preds, dtype=int)
        
        balanced_acc = balanced_accuracy_score(all_targets_array, all_preds_array)
        f1_macro = f1_score(all_targets_array, all_preds_array, average='macro')
        f1_weighted = f1_score(all_targets_array, all_preds_array, average='weighted')
        mcc = matthews_corrcoef(all_targets_array, all_preds_array)
        
        # Calculate metrics with rejection logic
        if len(all_preds_with_rejection) > 0:
            all_preds_with_rejection_array = np.array(all_preds_with_rejection, dtype=int)
            f1_macro_with_rejection = f1_score(all_targets_array, all_preds_with_rejection_array, average='macro')
            mcc_with_rejection = matthews_corrcoef(all_targets_array, all_preds_with_rejection_array)
        else:
            f1_macro_with_rejection = f1_macro
            mcc_with_rejection = mcc
        
        # Multi-class metrics
        all_probs = np.array(all_probs)
        if len(np.unique(all_targets_array)) == 2:
            auc_score = roc_auc_score(all_targets_array, all_probs[:, 1])
            pr_auc = average_precision_score(all_targets_array, all_probs[:, 1])
        else:
            try:
                auc_score = roc_auc_score(all_targets_array, all_probs, multi_class='ovr')
                pr_auc = np.mean([
                    average_precision_score((all_targets_array == i).astype(int), all_probs[:, i])
                    for i in range(len(np.unique(all_targets_array)))
                ])
            except:
                auc_score = 0.0
                pr_auc = 0.0
        
        # Log class distribution
        unique_targets, target_counts = np.unique(all_targets_array, return_counts=True)
        class_distribution = dict(zip(unique_targets, target_counts))
        
        # Store logits and labels for temperature scaling
        self.val_logits = all_logits
        self.val_labels = all_targets_array
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_balanced_accuracy': balanced_acc,
            'val_f1_macro': f1_macro,
            'val_f1_macro_with_rejection': f1_macro_with_rejection,
            'val_f1_weighted': f1_weighted,
            'val_mcc': mcc,
            'val_mcc_with_rejection': mcc_with_rejection,
            'val_auc': auc_score,
            'val_pr_auc': pr_auc,
            'val_class_distribution': class_distribution
        }
        
        self.val_metrics.append(metrics)
        return metrics
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, Any], epoch: int):
        """Log metrics to tensorboard and console"""
        
        # Console logging
        logger.info(f"Epoch {epoch + 1}/{self.config.training.epochs}")
        logger.info(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
                f"Acc: {train_metrics['train_accuracy']:.4f}, "
                f"F1: {train_metrics['train_f1_macro']:.4f}")
        logger.info(f"Val - Loss: {val_metrics['val_loss']:.4f}, "
                f"Acc: {val_metrics['val_accuracy']:.4f}, "
                f"F1: {val_metrics['val_f1_macro']:.4f}, "
                f"MCC: {val_metrics['val_mcc']:.4f}")
        
        # Tensorboard logging
        if self.writer:
            # Training metrics
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Training/{key}', value, epoch)
            
            # Validation metrics (can contain non-scalars)
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Validation/{key}', value, epoch)
                elif isinstance(value, dict):
                    # For dictionaries, log as text, not as a scalar
                    self.writer.add_text(f'Validation/{key}', str(value), epoch)
            
            # Learning rate
            if self.optimizer:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement"""
        if self.config.training.monitor_mode == "max":
            return current_score > self.best_score
        else:
            return current_score < self.best_score
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        if not self.config.save_best_only or self.config.checkpoint_dir:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_score': self.best_score,
                'config': self.config
            }
            
            if self.scheduler is not None:
                try:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                except Exception as e:
                    logger.warning(f"Could not save scheduler state: {e}")
            
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            torch.save(checkpoint, checkpoint_dir / filename)
    
    def _get_device(self) -> torch.device:
        """Get appropriate device"""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device)
    
    def _fit_temperature_scaling(self):
        """Fit temperature scaling parameter on validation set"""
        try:
            # Convert to tensors
            val_logits = torch.tensor(self.val_logits, device=self.device)
            val_labels = torch.tensor(self.val_labels, device=self.device)
            
            # Initialize temperature parameter
            temperature = nn.Parameter(torch.ones(1) * 1.5, device=self.device)
            optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
            
            def eval():
                optimizer.zero_grad()
                loss = F.cross_entropy(val_logits / temperature, val_labels)
                loss.backward()
                return loss
            
            optimizer.step(eval)
            
            # Clamp temperature to reasonable range
            self.temperature = max(0.5, min(5.0, temperature.item()))
            logger.info(f"Temperature scaling fitted: T = {self.temperature:.3f}")
            
        except Exception as e:
            logger.warning(f"Temperature scaling failed: {e}, using default T=1.0")
            self.temperature = 1.0
    
    def _apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        return logits / self.temperature

# =========================
# ENHANCED CROSS VALIDATION
# =========================

class TimeSeriesCV:
    """Enhanced time series cross-validation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series cross-validation splits"""
        n_samples = len(data)
        
        if self.config.validation_method == "walk_forward":
            return self._walk_forward_splits(n_samples)
        elif self.config.validation_method == "expanding_window":
            return self._expanding_window_splits(n_samples)
        else:
            raise ValueError(f"Unknown validation method: {self.config.validation_method}")
    
    def _walk_forward_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Walk-forward validation splits"""
        splits = []
        
        # Calculate split sizes
        train_size = int(n_samples * self.config.train_ratio)
        val_size = int(n_samples * self.config.val_ratio)
        
        # Create multiple splits
        for i in range(self.config.n_splits):
            start_idx = i * val_size // 2  # Overlap between splits
            train_end = start_idx + train_size
            val_start = train_end + self.config.embargo_period
            val_end = val_start + val_size
            
            if val_end <= n_samples:
                train_indices = np.arange(start_idx, train_end)
                val_indices = np.arange(val_start, val_end)
                splits.append((train_indices, val_indices))
        
        return splits
    
    def _expanding_window_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Expanding window validation splits"""
        splits = []
        
        # Calculate initial sizes
        initial_train_size = int(n_samples * self.config.train_ratio)
        val_size = int(n_samples * self.config.val_ratio)
        
        for i in range(self.config.n_splits):
            # Expand training window
            train_size = initial_train_size + i * val_size // 2
            val_start = train_size + self.config.embargo_period
            val_end = val_start + val_size
            
            if val_end <= n_samples:
                train_indices = np.arange(0, train_size)
                val_indices = np.arange(val_start, val_end)
                splits.append((train_indices, val_indices))
        
        return splits

# =========================
# MAIN PIPELINE
# =========================

class OptimizedStockPredictor:
    """Main optimized stock prediction pipeline"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.data_processor = OptimizedDataProcessor(config.data)
        self.cv = TimeSeriesCV(config.data)
        
        # Set seeds for reproducibility
        self._set_seeds()
        
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the complete pipeline"""
        logger.info("Starting optimized stock prediction pipeline")
        
        # Data processing
        logger.info("Processing data...")
        df_processed = self.data_processor.fit_transform(df)
        
        # Validate processed data
        if df_processed is None:
            raise ValueError("Data processing returned None")
        
        if len(df_processed) < self.config.data.window_size * 10:
            raise ValueError("Insufficient data for training")
        
        # Prepare features and targets
        # Exclude datetime columns and ensure only numeric features are used
        feature_cols = []
        for col in df_processed.columns:
            if col != 'target':
                # Check if column is numeric
                if df_processed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    feature_cols.append(col)
                else:
                    logger.warning(f"Excluding non-numeric column '{col}' with dtype '{df_processed[col].dtype}' from features")
        
        if len(feature_cols) == 0:
            raise ValueError("No numeric features found after processing")
        
        X = df_processed[feature_cols].values
        y = df_processed['target'].values
        
        # Final safety check: ensure X is numeric
        if not np.issubdtype(X.dtype, np.number):
            logger.warning(f"Converting feature matrix from {X.dtype} to float32")
            X = X.astype(np.float32)
        
        # Ensure y is integer
        if not np.issubdtype(y.dtype, np.integer):
            logger.warning(f"Converting target vector from {y.dtype} to int64")
            y = y.astype(np.int64)
        
        # Safety check for data integrity
        if X is None or y is None:
            raise ValueError("Features or targets contain None values after processing")
        
        if np.any(safe_isnan(X)) or np.any(safe_isnan(y)):
            logger.warning("NaN values detected, cleaning data...")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
        
        logger.info(f"Data processed: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Cross-validation
        cv_results = []
        splits = self.cv.split(df_processed)
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Training fold {fold + 1}/{len(splits)}")
            
            # Split data
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Validate split data
            if (X_train is None or y_train is None or X_val is None or y_val is None or
                len(X_train) == 0 or len(X_val) == 0):
                logger.error(f"Invalid data split for fold {fold + 1}")
                continue
            
            # Per-fold feature selection and scaling to avoid data leakage
            X_train_processed, X_val_processed = self._process_fold_features(X_train, y_train, X_val)
            
            # Train model
            try:
                trainer = OptimizedTrainer(self.config)
                fold_results = trainer.fit(X_train_processed, y_train, X_val_processed, y_val)
                fold_results['fold'] = fold
                cv_results.append(fold_results)
            except Exception as e:
                logger.error(f"Training failed for fold {fold + 1}: {e}")
                continue
            
            # Memory cleanup
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Validate results
        if not cv_results:
            raise ValueError("No successful training folds completed")
        
        # Aggregate results
        self._log_cv_results(cv_results)
        
        return {
            'cv_results': cv_results,
            'config': self.config,
            'data_processor': self.data_processor
        }
    
    def _process_fold_features(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process features per-fold to avoid data leakage"""
        # Feature selection
        if X_train.shape[1] > self.config.data.max_features:
            X_train_selected, X_val_selected = self._select_features_per_fold(X_train, y_train, X_val)
        else:
            X_train_selected, X_val_selected = X_train, X_val
        
        # Feature scaling
        X_train_scaled, X_val_scaled = self._scale_features_per_fold(X_train_selected, X_val_selected)
        
        return X_train_scaled, X_val_scaled
    
    def _select_features_per_fold(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select features using only training data"""
        # Ensure data is numeric and handle any remaining non-numeric values
        if not np.issubdtype(X_train.dtype, np.number):
            logger.warning(f"Converting non-numeric training data from {X_train.dtype} to float32")
            X_train = X_train.astype(np.float32)
        if not np.issubdtype(X_val.dtype, np.number):
            logger.warning(f"Converting non-numeric validation data from {X_val.dtype} to float32")
            X_val = X_val.astype(np.float32)
        
        # Handle any remaining NaN or infinite values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.config.data.feature_selection_method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=min(self.config.data.max_features, X_train.shape[1]))
        elif self.config.data.feature_selection_method == "f_score":
            selector = SelectKBest(score_func=f_classif, k=min(self.config.data.max_features, X_train.shape[1]))
        else:
            # Just use correlation filtering
            X_train_filtered, keep_indices = self._remove_correlated_features(X_train, list(range(X_train.shape[1])))
            X_val_filtered = X_val[:, keep_indices]
            return X_train_filtered, X_val_filtered
        
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        
        logger.info(f"Fold feature selection: {X_train.shape[1]} -> {X_train_selected.shape[1]} features")
        return X_train_selected, X_val_selected
    
    def _scale_features_per_fold(self, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using only training data statistics"""
        # Ensure data is numeric and handle any remaining non-numeric values
        if not np.issubdtype(X_train.dtype, np.number):
            logger.warning(f"Converting non-numeric training data from {X_train.dtype} to float32")
            X_train = X_train.astype(np.float32)
        if not np.issubdtype(X_val.dtype, np.number):
            logger.warning(f"Converting non-numeric validation data from {X_val.dtype} to float32")
            X_val = X_val.astype(np.float32)
        
        # Handle any remaining NaN or infinite values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        # If global scaler is disabled, return original data (window normalization will be applied)
        if not self.config.data.use_global_scaler:
            logger.info("Global scaler disabled: using only window normalization")
            return X_train, X_val
        
        # Apply global scaling
        # CORRECTED THE PATH FROM self.config.scaler_type TO self.config.data.scaler_type
        if self.config.data.scaler_type == "robust":
            scaler = RobustScaler()
        elif self.config.data.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.config.data.scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.data.scaler_type}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        return X_train_scaled, X_val_scaled
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Set to True for better performance with fixed input sizes
    
    def _log_cv_results(self, cv_results: List[Dict[str, Any]]):
        """Log cross-validation results"""
        logger.info("=" * 60)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("=" * 60)
        
        # Extract scores
        scores = [result['best_score'] for result in cv_results]
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        logger.info(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
        logger.info(f"Individual fold scores: {[f'{score:.4f}' for score in scores]}")
        
        # Calculate confidence interval
        from scipy import stats
        confidence_interval = stats.t.interval(
            0.95, len(scores) - 1, loc=mean_score, 
            scale=stats.sem(scores)
        )
        logger.info(f"95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        
        logger.info("=" * 60)

# =========================
# HYPERPARAMETER OPTIMIZATION
# =========================

def optimize_hyperparameters(df: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna"""
    
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available, skipping hyperparameter optimization")
        return {}
    
    def objective(trial):
        # Suggest hyperparameters
        config = OptimizedConfig()
        
        # Model hyperparameters
        config.model.base_channels = trial.suggest_categorical('base_channels', [16, 32, 64])
        config.model.num_blocks = trial.suggest_int('num_blocks', 2, 5)
        config.model.dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        config.model.attention_type = trial.suggest_categorical('attention_type', ['se', 'cbam', 'eca'])
        
        # Training hyperparameters
        config.training.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        config.training.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        config.training.focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
        
        # Data hyperparameters
        config.data.window_size = trial.suggest_categorical('window_size', [30, 60, 90])
        config.data.max_features = trial.suggest_int('max_features', 20, 100)
        
        # Quick training for optimization
        config.training.epochs = 50
        config.training.early_stopping_patience = 10
        
        # Ensure stable scheduler for optimization trials
        config.training.scheduler_type = "cosine_warm"
        
        # Debug: Log suggested hyperparameters
        logger.info(f"Trial hyperparameters: base_channels={config.model.base_channels}, "
                   f"num_blocks={config.model.num_blocks}, dropout_rate={config.model.dropout_rate}, "
                   f"attention_type={config.model.attention_type}, learning_rate={config.training.learning_rate}, "
                   f"weight_decay={config.training.weight_decay}, focal_gamma={config.training.focal_gamma}, "
                   f"window_size={config.data.window_size}, max_features={config.data.max_features}")
        
        # Validate configuration before training
        try:
            # Validate model configuration
            if config.model.base_channels < 1:
                raise ValueError(f"Invalid base_channels: {config.model.base_channels}")
            if config.model.num_blocks < 1:
                raise ValueError(f"Invalid num_blocks: {config.model.num_blocks}")
            if config.model.dropout_rate < 0 or config.model.dropout_rate > 1:
                raise ValueError(f"Invalid dropout_rate: {config.model.dropout_rate}")
            if config.model.attention_type not in ['se', 'cbam', 'eca']:
                raise ValueError(f"Invalid attention_type: {config.model.attention_type}")
            
            # Validate training configuration
            if config.training.learning_rate <= 0:
                raise ValueError(f"Invalid learning_rate: {config.training.learning_rate}")
            if config.training.weight_decay < 0:
                raise ValueError(f"Invalid weight_decay: {config.training.weight_decay}")
            if config.training.focal_gamma < 0:
                raise ValueError(f"Invalid focal_gamma: {config.training.focal_gamma}")
            
            # Validate data configuration
            if config.data.window_size < 1:
                raise ValueError(f"Invalid window_size: {config.data.window_size}")
            if config.data.max_features < 1:
                raise ValueError(f"Invalid max_features: {config.data.max_features}")
            
            # Train model
            predictor = OptimizedStockPredictor(config)
            results = predictor.fit(df)
            
            # Return average CV score
            cv_scores = [result['best_score'] for result in results['cv_results']]
            return np.mean(cv_scores)
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Hyperparameter optimization completed")
    logger.info(f"Best score: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }

# =========================
# MAIN EXECUTION FUNCTION
# =========================

def main():
    """Main execution function"""
    logger.info("Starting Optimized Stock CNN Pipeline")
    
    # Configuration
    config = OptimizedConfig()
    
    # Advanced configuration for expert users
    config.model.model_type = "enhanced_cnn"
    config.model.base_channels = 16  # Reduced for lighter model
    config.model.num_blocks = 2  # Reduced for lighter model
    config.model.attention_type = "se"  # Lighter attention
    config.model.dropout_rate = 0.4  # Increased regularization
    config.model.norm_type = "group"  # Use GroupNorm for small batches
    
    config.training.optimizer_type = "adamw"
    config.training.scheduler_type = "cosine_warm"  # Start with stable scheduler
    config.training.loss_type = "label_smoothing"  # Better calibration
    config.training.use_amp = True
    config.training.epochs = 200
    config.training.early_stopping_patience = 25
    config.training.monitor_metric = "val_mcc"  # More stable metric
    
    config.data.feature_engineering_level = "expert"
    config.data.labeling_method = "volatility_adaptive"
    config.data.window_size = 60
    config.data.max_features = 50
    config.data.embargo_period = 65  # window_size + prediction_horizon
    
    logger.info(f"Configuration: {config}")
    
    # Load data
    try:
        # Replace with your actual data path
        file_path = r"F:\VesperSet\stock_data_analysis\data\raw\indexData\HS300_daily_kline.csv"
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        # Create sample data for demonstration
        logger.info("Creating sample data for demonstration")
        dates = pd.date_range('2020-01-01', periods=2000, freq='D')
        np.random.seed(42)
        
        # Generate realistic stock data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.lognormal(15, 0.5, len(dates)).astype(int)
        })
        logger.info("Sample data created")
    
    try:
        # Hyperparameter optimization (optional)
        if OPTUNA_AVAILABLE:
            logger.info("Running hyperparameter optimization...")
            optimization_results = optimize_hyperparameters(df, n_trials=20)
            
            # Update config with best parameters
            if optimization_results and 'best_params' in optimization_results:
                best_params = optimization_results['best_params']
                for key, value in best_params.items():
                    if hasattr(config.model, key):
                        setattr(config.model, key, value)
                    elif hasattr(config.training, key):
                        setattr(config.training, key, value)
                    elif hasattr(config.data, key):
                        setattr(config.data, key, value)
                logger.info("Configuration updated with optimized hyperparameters")
        
        # Train final model
        logger.info("Training final model with optimized configuration...")
        predictor = OptimizedStockPredictor(config)
        results = predictor.fit(df)
        
        # Performance analysis
        cv_scores = [result['best_score'] for result in results['cv_results']]
        logger.success(f"Training completed successfully!")
        logger.success(f"Final CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Save results
        results_path = Path("results")
        results_path.mkdir(exist_ok=True)
        
        import pickle
        with open(results_path / "training_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

# =========================
# TEST FUNCTION
# =========================

def test_model_initialization():
    """Test if the model can be initialized without errors"""
    try:
        # Create a simple configuration
        config = OptimizedConfig()
        
        # Test model initialization
        model = OptimizedStockCNN(num_classes=3, config=config.model)
        print("✓ Model initialization successful")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 1, 30, 50)  # batch_size=1, channels=1, height=30, width=50
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test model initialization first
    print("Testing model initialization...")
    if test_model_initialization():
        print("Model test passed, proceeding with main execution...")
        main()
    else:
        print("Model test failed, please check the error above")