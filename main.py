# financial_time_series_model.py
import os
import glob
import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from loguru import logger
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== CONSTANTS ====================
# Data constants
DEFAULT_WINDOW_SIZE = 60
DEFAULT_PREDICTION_HORIZON = 5
DEFAULT_RETURN_THRESHOLD = 0.015
MIN_STOCK_DATA_POINTS = 100
SMALL_VALUE = 1e-8  # 防止除零

# Model constants
DEFAULT_DROPOUT = 0.2
CONV_KERNEL_SIZES = {'conv1': 7, 'conv2': 5, 'conv3': 3, 'res1': 5, 'res2': 3}
HIDDEN_DIMS = {'conv1': 64, 'conv2': 128, 'conv3': 256, 'fc1': 256, 'fc2': 128}

# Training constants
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 30
GRADIENT_CLIP_VALUE = 1.0
EARLY_STOPPING_PATIENCE = 8
LR_SCHEDULER_PATIENCE = 4
LR_SCHEDULER_FACTOR = 0.5
SEED = 42

# Focal loss constants
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 2.0

# RSI constants
RSI_WINDOW = 14

# Rolling window sizes for indicators
ROLLING_WINDOWS = {
    'ma_short': 5,
    'ma_medium': 20,
    'ma_long': 50,
    'vol_short': 5,
    'vol_medium': 20,
    'momentum': 10,
    'volume_ma': 10,
    'price_position': 20,
    'turnover_ma': 5,
    'change_pct_vol': 10,
    'trans_amt_ma': 10
}

# Class imbalance thresholds
SEVERE_IMBALANCE_THRESHOLD_LOW = 0.1
SEVERE_IMBALANCE_THRESHOLD_HIGH = 0.9

# Required and optional columns
REQUIRED_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']
OPTIONAL_COLUMNS = ['secu_code', 'transaction_amount', 'amplitude',
                    'turnover_rate', 'change_percentage', 'change_amount']
DATE_COLUMN_NAMES = ['trade_date', 'date']

# Logging
LOG_FILE = "training.log"
os.makedirs("outputs", exist_ok=True)
logger.add(LOG_FILE, rotation="5 MB", retention="10 days", encoding="utf-8")
logger.info("Logger initialized")


# ==================== UTILS ====================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_warmup_bars():
    return max(ROLLING_WINDOWS.values())


def safe_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(method='ffill').fillna(0)


def load_csv_any_datecol(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col_found = False
        for date_col in DATE_COLUMN_NAMES:
            if date_col in df.columns:
                df['date'] = pd.to_datetime(df[date_col])
                date_col_found = True
                break
        if not date_col_found and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']); date_col_found = True
        if not date_col_found:
            raise ValueError(f"No date column found in {path}")
        df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
        return df
    except Exception as e:
        logger.exception(f"Error load_csv_any_datecol {path}: {e}")
        return None


# ==================== STOCK DATA & INDICATORS ====================
def load_kline(csv_path):
    try:
        df = load_csv_any_datecol(csv_path)
        if df is None:
            return None
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        available_extra = [c for c in OPTIONAL_COLUMNS if c in df.columns]
        ret_cols = REQUIRED_COLUMNS + available_extra
        if 'secu_code' not in df.columns:
            code = os.path.splitext(os.path.basename(csv_path))[0]
            df['secu_code'] = str(code)
            ret_cols = ['secu_code'] + ret_cols
        return df[ret_cols]
    except Exception as e:
        logger.exception(f"Error loading {csv_path}: {e}")
        return None


def calculate_rsi(prices, window=RSI_WINDOW):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + SMALL_VALUE)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def robust_winsorize(s: pd.Series, lower_q=0.005, upper_q=0.995) -> pd.Series:
    """时间序列友好的去极值：按分位数裁剪（逐列），避免引入未来信息。"""
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def add_indicators_single_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗 + 指标生成（单股内）：
    - 去重、排序
    - 基础合理性检查：非正价格、体量异常归零；0成交量天保留但特征做处理
    - 稳健去极值：对收益/变动类做 winsor
    - 仅前向填充，避免未来泄漏
    """
    df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True).copy()

    # --- 数据质量检查 ---
    for col in ['open','high','low','close','volume']:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")
    # 非正价格处理（置为NaN，后续由ffill处理）
    for col in ['open','high','low','close']:
        df.loc[df[col] <= 0, col] = np.nan
    # 成交量异常（负数）置0
    df.loc[df['volume'] < 0, 'volume'] = 0
    # 极端跳变（可能是拆分/复权问题）可按需标记；这里不直接删除，以免误伤

    # --- 指标 ---
    df['ret1_raw'] = df['close'].pct_change()
    df['ret1'] = robust_winsorize(df['ret1_raw'])
    df['ret5'] = robust_winsorize(df['close'].pct_change(ROLLING_WINDOWS['ma_short']))
    df['ma5'] = df['close'].rolling(ROLLING_WINDOWS['ma_short']).mean()
    df['ma20'] = df['close'].rolling(ROLLING_WINDOWS['ma_medium']).mean()
    df['ma50'] = df['close'].rolling(ROLLING_WINDOWS['ma_long']).mean()

    df['mom10'] = robust_winsorize(df['close'].pct_change(ROLLING_WINDOWS['momentum']))
    df['rsi'] = calculate_rsi(df['close'], RSI_WINDOW)

    df['vol5'] = df['ret1'].rolling(ROLLING_WINDOWS['vol_short']).std()
    df['vol20'] = df['ret1'].rolling(ROLLING_WINDOWS['vol_medium']).std()

    df['vol_ma10'] = df['volume'].rolling(ROLLING_WINDOWS['volume_ma']).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma10'] + SMALL_VALUE)

    wpp = ROLLING_WINDOWS['price_position']
    df['price_pos'] = (df['close'] - df['low'].rolling(wpp).min()) / \
                      (df['high'].rolling(wpp).max() - df['low'].rolling(wpp).min() + SMALL_VALUE)

    df['hl_ratio'] = (df['high'] - df['low']) / (df['close'] + SMALL_VALUE)
    df['oc_ratio'] = (df['close'] - df['open']) / (df['open'] + SMALL_VALUE)

    if 'amplitude' in df.columns:
        df['amplitude_norm'] = robust_winsorize(df['amplitude'] / 100.0)
    if 'turnover_rate' in df.columns:
        df['turnover_ma5'] = df['turnover_rate'].rolling(ROLLING_WINDOWS['turnover_ma']).mean()
        df['turnover_ratio'] = df['turnover_rate'] / (df['turnover_ma5'] + SMALL_VALUE)
    if 'change_percentage' in df.columns:
        df['change_pct_ma5'] = df['change_percentage'].rolling(ROLLING_WINDOWS['ma_short']).mean()
        df['change_pct_vol'] = df['change_percentage'].rolling(ROLLING_WINDOWS['change_pct_vol']).std()
    if 'transaction_amount' in df.columns:
        df['trans_amt_ma10'] = df['transaction_amount'].rolling(ROLLING_WINDOWS['trans_amt_ma']).mean()
        df['trans_amt_ratio'] = df['transaction_amount'] / (df['trans_amt_ma10'] + SMALL_VALUE)
    if 'change_amount' in df.columns and 'volume' in df.columns:
        df['amt_vol_ratio'] = df['change_amount'] / (df['volume'] + SMALL_VALUE)

    df = safe_forward_fill(df)
    df = df.replace([np.inf, -np.inf], 0)
    return df


def make_label_single_stock(df, H=DEFAULT_PREDICTION_HORIZON, tau=DEFAULT_RETURN_THRESHOLD):
    future_price = df['close'].shift(-H)
    future_return = (future_price - df['close']) / (df['close'] + SMALL_VALUE)
    labels = (future_return > tau).astype(int)
    return labels


def get_feature_columns():
    base_features = [
        'open', 'high', 'low', 'close', 'volume', 'ret1', 'ret5',
        'ma5', 'ma20', 'ma50', 'mom10', 'rsi', 'vol5', 'vol20',
        'vol_ma10', 'vol_ratio', 'price_pos', 'hl_ratio', 'oc_ratio'
    ]
    optional_features = [
        'amplitude_norm', 'turnover_ma5', 'turnover_ratio',
        'change_pct_ma5', 'change_pct_vol', 'trans_amt_ma10',
        'trans_amt_ratio', 'amt_vol_ratio'
    ]
    return base_features, optional_features


# ==================== INDEX / SECTOR FEATURES (OPTIONAL) ====================
def add_index_indicators(df_idx: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df_idx[['date', 'close']].copy()
    df[f'{prefix}ret1'] = robust_winsorize(df['close'].pct_change())
    df[f'{prefix}ma20'] = df['close'].rolling(20).mean()
    df[f'{prefix}vol20'] = df[f'{prefix}ret1'].rolling(20).std()
    df[f'{prefix}rsi'] = calculate_rsi(df['close'], 14)
    df = safe_forward_fill(df)
    return df.drop(columns=['close'])


def merge_index_features(stock_df: pd.DataFrame,
                         mkt_index_df: Optional[pd.DataFrame],
                         sector_index_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = stock_df.copy()
    if mkt_index_df is not None:
        df = df.merge(mkt_index_df, on='date', how='left')
    if sector_index_df is not None:
        df = df.merge(sector_index_df, on='date', how='left')
    df = safe_forward_fill(df)
    return df


def load_market_index_features(mkt_index_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not mkt_index_path:
        return None
    idx = load_csv_any_datecol(mkt_index_path)
    if idx is None or 'close' not in idx.columns:
        logger.warning(f"Market index file invalid: {mkt_index_path}")
        return None
    return add_index_indicators(idx, prefix='mkt_')


def load_sector_index_features(sector_index_dir: Optional[str], sector_code: Optional[str]) -> Optional[pd.DataFrame]:
    if not sector_index_dir or not sector_code:
        return None
    f = os.path.join(sector_index_dir, f"{sector_code}.csv")
    if not os.path.exists(f):
        return None
    idx = load_csv_any_datecol(f)
    if idx is None or 'close' not in idx.columns:
        return None
    return add_index_indicators(idx, prefix='sec_')


def load_sector_map(sector_map_csv: Optional[str]) -> Dict[str, str]:
    if not sector_map_csv or not os.path.exists(sector_map_csv):
        return {}
    df = pd.read_csv(sector_map_csv)
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    if 'secu_code' not in cols or 'sector_code' not in cols:
        return {}
    return {str(r['secu_code']): str(r['sector_code']) for _, r in df.iterrows()}


def extend_feature_columns_with_index(base_feats: List[str]) -> List[str]:
    idx_feats = ['mkt_ret1', 'mkt_ma20', 'mkt_vol20', 'mkt_rsi']
    sec_feats = ['sec_ret1', 'sec_ma20', 'sec_vol20', 'sec_rsi']
    return base_feats + idx_feats + sec_feats


# ==================== DATASET ====================
class Kline1DDataset(Dataset):
    def __init__(self, folder, window=DEFAULT_WINDOW_SIZE, H=DEFAULT_PREDICTION_HORIZON,
                 tau=DEFAULT_RETURN_THRESHOLD, file_glob="*.csv",
                 split=("2015-01-01", "2023-12-31"), normalize=True, shared_scaler=None,
                 mkt_index_path: Optional[str] = None,
                 sector_index_dir: Optional[str] = None,
                 sector_map_csv: Optional[str] = None):

        self.window = window
        self.H = H
        self.normalize = normalize
        self.samples = []
        self.sample_meta = []  # (secu_code, stock_name, t, date)
        self.scaler = shared_scaler

        logger.info(f"Loading data from {folder} with split {split}")
        csv_files = glob.glob(os.path.join(folder, file_glob))
        if not csv_files:
            raise ValueError(f"No CSV files found in {folder} matching {file_glob}")
        logger.info(f"Found {len(csv_files)} CSV files")

        mkt_idx_feat = load_market_index_features(mkt_index_path)
        sector_map = load_sector_map(sector_map_csv)
        warmup_bars = get_warmup_bars()

        all_train_features = []
        all_stock_data = []

        base_features, optional_features = get_feature_columns()
        ext_base_features = extend_feature_columns_with_index(base_features)

        for fp in csv_files:
            try:
                df = load_kline(fp)
                if df is None or len(df) < MIN_STOCK_DATA_POINTS:
                    continue
                secu_code = str(df['secu_code'].iloc[0]) if 'secu_code' in df.columns else os.path.splitext(os.path.basename(fp))[0]
                sector_code = sector_map.get(secu_code, None)
                sec_idx_feat = load_sector_index_features(sector_index_dir, sector_code)

                df = add_indicators_single_stock(df)
                df = merge_index_features(df, mkt_idx_feat, sec_idx_feat)

                labels = make_label_single_stock(df, H, tau)
                df['label'] = labels

                feat_cols = [col for col in ext_base_features if col in df.columns]
                feat_cols.extend([col for col in optional_features if col in df.columns])

                train_mask = (df['date'] >= pd.Timestamp(split[0])) & (df['date'] <= pd.Timestamp(split[1]))
                train_df = df[train_mask]
                if len(train_df) > max(window, warmup_bars) + H:
                    train_features = train_df[feat_cols].values.astype(np.float32)
                    all_train_features.append(train_features)

                all_stock_data.append({
                    'df': df.reset_index(drop=True),
                    'feat_cols': feat_cols,
                    'stock_name': os.path.basename(fp),
                    'secu_code': secu_code
                })
            except Exception as e:
                logger.exception(f"Process file failed: {fp} | {e}")

        if not all_train_features:
            raise ValueError("No valid training data found")

        if self.normalize and self.scaler is None:
            logger.info("Fitting scaler on training data only...")
            train_features_combined = np.vstack(all_train_features)
            self.scaler = StandardScaler()
            self.scaler.fit(train_features_combined)
            logger.info(f"Scaler fitted on {train_features_combined.shape[0]} rows")

        valid_samples = 0
        label_counts = {0: 0, 1: 0}
        for stock_data in all_stock_data:
            df = stock_data['df']
            feat_cols = stock_data['feat_cols']
            secu_code = stock_data['secu_code']
            stock_name = stock_data['stock_name']

            X = df[feat_cols].values.astype(np.float32)
            if self.normalize and self.scaler is not None:
                X = self.scaler.transform(X)

            for t in range(max(self.window, warmup_bars), len(df) - self.H):
                date_end = pd.Timestamp(df.iloc[t]['date'])
                if not (pd.Timestamp(split[0]) <= date_end <= pd.Timestamp(split[1])):
                    continue
                label = df.iloc[t]['label']
                if pd.isna(label):
                    continue
                x_win = X[t - self.window:t].T  # [C,T]
                self.samples.append((torch.from_numpy(x_win.copy()), int(label)))
                self.sample_meta.append((secu_code, stock_name, t, pd.Timestamp(df.iloc[t]['date'])))
                valid_samples += 1
                label_counts[int(label)] += 1

        logger.info(f"Created {valid_samples} samples from {len(all_stock_data)} stocks")
        if len(self.samples) == 0:
            raise ValueError("No valid samples created. Check parameters and data quality.")

        pos_ratio = label_counts[1] / sum(label_counts.values()) if sum(label_counts.values()) else 0.0
        logger.info(f"Class distribution: {label_counts}; Positive ratio: {pos_ratio:.3f}")
        if pos_ratio < SEVERE_IMBALANCE_THRESHOLD_LOW or pos_ratio > SEVERE_IMBALANCE_THRESHOLD_HIGH:
            logger.warning(f"Severe class imbalance! Consider adjusting tau.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def meta(self, idx):
        return self.sample_meta[idx]

    def get_scaler(self):
        return self.scaler


def collate_fn(batch):
    x_batch = torch.stack([sample[0] for sample in batch])
    y_batch = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    return x_batch, y_batch


# ==================== MODELS ====================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=5, padding=2, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.activation(x); x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, kernel_size // 2, dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.conv2(out); out = self.bn(out)
        out += residual; out = F.gelu(out); out = self.dropout(out)
        return out


class TrendCNN1D(nn.Module):
    def __init__(self, in_channels, n_classes=2, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, HIDDEN_DIMS['conv1'], CONV_KERNEL_SIZES['conv1'],
                               CONV_KERNEL_SIZES['conv1']//2, dropout)
        self.conv2 = ConvBlock(HIDDEN_DIMS['conv1'], HIDDEN_DIMS['conv2'], CONV_KERNEL_SIZES['conv2'],
                               CONV_KERNEL_SIZES['conv2']//2, dropout)
        self.conv3 = ConvBlock(HIDDEN_DIMS['conv2'], HIDDEN_DIMS['conv3'], CONV_KERNEL_SIZES['conv3'],
                               CONV_KERNEL_SIZES['conv3']//2, dropout)
        self.res1 = ResidualBlock(HIDDEN_DIMS['conv2'], CONV_KERNEL_SIZES['res1'], dropout)
        self.res2 = ResidualBlock(HIDDEN_DIMS['conv3'], CONV_KERNEL_SIZES['res2'], dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIMS['conv3']*2, HIDDEN_DIMS['fc1']),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIMS['fc1'], HIDDEN_DIMS['fc2']),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIMS['fc2'], n_classes)
        )
    def forward(self, x):
        x = self.conv1(x); x = F.max_pool1d(x, 2)
        x = self.conv2(x); x = self.res1(x); x = F.max_pool1d(x, 2)
        x = self.conv3(x); x = self.res2(x)
        avg_pool = self.global_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(x)


class TrendLSTM(nn.Module):
    """
    LSTM 对比模型：输入 [B,C,T] => 转为 [B,T,C] 喂给 LSTM
    """
    def __init__(self, in_channels, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=True, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = x.transpose(1, 2)  # [B,T,C]
        out, _ = self.lstm(x)  # [B,T,H*dir]
        # 取最后时刻，也可试 mean-pool over time
        last = out[:, -1, :]
        return self.head(last)


# ==================== TRAIN / EVAL ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0; total_samples = 0; correct = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total_loss / total_samples, correct / total_samples


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0; total_samples = 0; correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            all_preds.extend(pred.cpu().numpy()); all_labels.extend(y.cpu().numpy())
    return total_loss / total_samples, correct / total_samples, all_preds, all_labels


def predict_proba(model: nn.Module, loader: DataLoader, device: str, mc_dropout_passes: int = 1):
    model.eval()
    probs_all = []
    if mc_dropout_passes <= 1:
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(device)
                logits = model(X)
                probs = torch.softmax(logits, dim=1)[:, 1]
                probs_all.append(probs.detach().cpu().numpy())
        return np.concatenate(probs_all)
    else:
        model.train()
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(device)
                acc = 0.0
                for _ in range(mc_dropout_passes):
                    logits = model(X)
                    acc += torch.softmax(logits, dim=1)[:, 1]
                probs_all.append((acc / mc_dropout_passes).detach().cpu().numpy())
        model.eval()
        return np.concatenate(probs_all)


def calculate_class_weights(dataset):
    labels = [sample[1] for sample in dataset.samples]
    unique, counts = np.unique(labels, return_counts=True)
    total = sum(counts)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)


# ==================== BACKTEST UTILS（同前版本，略） ====================
def equity_curve_from_trades(trade_rs, initial_capital=1.0):
    if trade_rs.size == 0: return np.array([initial_capital])
    return initial_capital * np.cumprod(1.0 + trade_rs)

def max_drawdown(equity):
    peak = -np.inf; mdd = 0.0; dd_dur = 0; max_dd_dur = 0
    for v in equity:
        if v > peak: peak = v; dd_dur = 0
        else:
            dd = 1.0 - v / (peak + SMALL_VALUE)
            mdd = max(mdd, dd); dd_dur += 1; max_dd_dur = max(max_dd_dur, dd_dur)
    return mdd, max_dd_dur

def annualize_return(total_ret, periods_per_year, n_periods):
    if n_periods == 0: return 0.0
    return (1.0 + total_ret) ** (periods_per_year / n_periods) - 1.0

def sharpe_ratio(ret_series, periods_per_year, rf=0.0):
    if ret_series.size == 0: return 0.0
    mean = np.mean(ret_series) - rf/periods_per_year
    vol  = np.std(ret_series, ddof=1)
    if vol <= 0: return 0.0
    return (mean * periods_per_year) / (vol * np.sqrt(periods_per_year))

def profit_stats(trade_rs):
    if trade_rs.size == 0:
        return dict(hit_rate=0.0, avg_win=0.0, avg_loss=0.0, win_loss_ratio=0.0, profit_factor=0.0)
    wins = trade_rs[trade_rs > 0]; losses = trade_rs[trade_rs <= 0]
    hit_rate = wins.size / trade_rs.size if trade_rs.size else 0.0
    avg_win  = wins.mean() if wins.size else 0.0
    avg_loss = -losses.mean() if losses.size else 0.0
    wl = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    pf = (wins.sum() / (-losses.sum())) if losses.size and -losses.sum() > 0 else np.inf
    return dict(hit_rate=float(hit_rate), avg_win=float(avg_win), avg_loss=float(avg_loss),
                win_loss_ratio=float(wl), profit_factor=float(pf))

def summarize_backtest(trade_rs, exposure_rate, turnover_count,
                       periods_per_year=252, holding_period=DEFAULT_PREDICTION_HORIZON):
    eq = equity_curve_from_trades(trade_rs)
    total_ret = eq[-1] / eq[0] - 1.0
    n_trades = trade_rs.size
    if n_trades > 1:
        day_rs = (1.0 + trade_rs) ** (1.0 / max(1, holding_period)) - 1.0
    else:
        day_rs = np.array([])
    ann_ret = annualize_return(total_ret, periods_per_year, n_trades * holding_period)
    ann_vol = np.std(day_rs, ddof=1) * np.sqrt(periods_per_year) if day_rs.size > 1 else 0.0
    sharpe  = sharpe_ratio(day_rs, periods_per_year) if day_rs.size > 1 else 0.0
    mdd, max_dd_dur = max_drawdown(eq)
    calmar = (ann_ret / mdd) if mdd > 0 else np.inf
    pstats = profit_stats(trade_rs)
    return {
        "trades": int(n_trades), "CAGR": float(ann_ret), "ann_vol": float(ann_vol),
        "Sharpe": float(sharpe), "maxDD": float(mdd), "Calmar": float(calmar),
        "hit_rate": pstats["hit_rate"], "avg_win": pstats["avg_win"], "avg_loss": pstats["avg_loss"],
        "win_loss_ratio": pstats["win_loss_ratio"], "profit_factor": pstats["profit_factor"],
        "exposure": float(exposure_rate), "turnover_count": int(turnover_count),
        "maxDD_duration_bars": int(max_dd_dur)
    }

def rebuild_indices_for_split(
    folder: str, file_glob: str, split: Tuple[str, str], window: int, H: int,
    scaler: Optional[StandardScaler],
    mkt_index_path: Optional[str],
    sector_index_dir: Optional[str],
    sector_map_csv: Optional[str]
):
    csv_files = glob.glob(os.path.join(folder, file_glob))
    warmup_bars = get_warmup_bars()
    df_list = []; idx_per_stock = []

    base_features, optional_features = get_feature_columns()
    ext_base_features = extend_feature_columns_with_index(base_features)
    mkt_idx_feat = load_market_index_features(mkt_index_path)
    sector_map = load_sector_map(sector_map_csv)

    for fp in csv_files:
        df = load_kline(fp)
        if df is None or len(df) < MIN_STOCK_DATA_POINTS:
            continue
        secu_code = str(df['secu_code'].iloc[0]) if 'secu_code' in df.columns else os.path.splitext(os.path.basename(fp))[0]
        sec_idx_feat = load_sector_index_features(sector_index_dir, sector_map.get(secu_code, None))
        df = add_indicators_single_stock(df)
        df = merge_index_features(df, mkt_idx_feat, sec_idx_feat)
        labels = make_label_single_stock(df, H, DEFAULT_RETURN_THRESHOLD)
        df['label'] = labels

        feat_cols = [col for col in ext_base_features if col in df.columns]
        feat_cols.extend([col for col in optional_features if col in df.columns])

        X = df[feat_cols].values.astype(np.float32)
        if scaler is not None:
            _ = scaler.transform(X)

        this_stock_indices = []
        for t in range(max(window, warmup_bars), len(df) - H):
            date_end = pd.Timestamp(df.iloc[t]['date'])
            if not (pd.Timestamp(split[0]) <= date_end <= pd.Timestamp(split[1])): continue
            if pd.isna(df.iloc[t]['label']): continue
            this_stock_indices.append(t)

        if this_stock_indices:
            df_list.append(df[['date', 'close']].reset_index(drop=True))
            idx_per_stock.append(this_stock_indices)
    return df_list, idx_per_stock

def backtest_long_flat_fixed_h(
    df_list: List[pd.DataFrame],
    indices_per_stock: List[List[int]],
    probas: np.ndarray,
    H=DEFAULT_PREDICTION_HORIZON,
    threshold=0.55,
    fee_rate=0.001
):
    assert len(probas) == sum(len(ix) for ix in indices_per_stock), "probas size mismatch"
    trade_rs = []; exposure_flags = []; turnover_count = 0
    ptr = 0
    for df, idx_list in zip(df_list, indices_per_stock):
        closes = df['close'].to_numpy()
        for i_end in idx_list:
            p = probas[ptr]; ptr += 1
            if p > threshold and (i_end + H) < len(closes):
                px_in  = closes[i_end]
                px_out = closes[i_end + H]
                ret_gross = (px_out - px_in) / (px_in + SMALL_VALUE)
                ret_net = ret_gross - 2.0 * fee_rate
                trade_rs.append(ret_net); exposure_flags.append(1); turnover_count += 1
            else:
                exposure_flags.append(0)
    trade_rs = np.array(trade_rs) if trade_rs else np.array([])
    exposure_rate = np.mean(exposure_flags) if exposure_flags else 0.0
    return trade_rs, exposure_rate, turnover_count

def backtest_daily_rebalance_timeseries(
    df_list: List[pd.DataFrame],
    idx_per_stock: List[List[int]],
    val_probas: np.ndarray,
    threshold: float = 0.55,
    H: int = DEFAULT_PREDICTION_HORIZON,
    fee_rate: float = 0.001
):
    all_dates = sorted(set(pd.concat([df['date'] for df in df_list]).unique()))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    daily_ret = np.zeros(len(all_dates), dtype=np.float64)
    daily_npos = np.zeros(len(all_dates), dtype=np.int32)
    ptr = 0
    for df, idxs in zip(df_list, idx_per_stock):
        closes = df['close'].to_numpy(); dates  = df['date'].to_numpy()
        for t in idxs:
            p = val_probas[ptr]; ptr += 1
            if p <= threshold: continue
            start = t + 1; end = min(t + H, len(closes) - 2)
            if start > end: continue
            fee_per_day = (2.0 * fee_rate) / max(1, H)
            for d in range(start, end + 1):
                r = (closes[d + 1] - closes[d]) / (closes[d] + SMALL_VALUE) - fee_per_day
                di = date_to_idx[pd.Timestamp(dates[d])]
                daily_ret[di] += r; daily_npos[di] += 1
    nz = daily_npos > 0
    daily_ret[nz] = daily_ret[nz] / daily_npos[nz]
    daily_ret[~nz] = 0.0
    return np.array(all_dates), daily_ret

def summarize_daily_timeseries(dates, daily_ret, rf=0.0, periods_per_year=252):
    eq = np.cumprod(1.0 + daily_ret)
    total_ret = eq[-1] - 1.0 if len(eq) > 0 else 0.0
    ann_ret = (1.0 + total_ret) ** (periods_per_year / max(1, len(daily_ret))) - 1.0 if len(daily_ret) > 0 else 0.0
    ann_vol = np.std(daily_ret, ddof=1) * np.sqrt(periods_per_year) if len(daily_ret) > 1 else 0.0
    sharpe = sharpe_ratio(daily_ret, periods_per_year, rf)
    mdd, max_dd_dur = max_drawdown(eq if len(eq)>0 else np.array([1.0]))
    calmar = (ann_ret / mdd) if mdd > 0 else np.inf
    turnover = np.count_nonzero(daily_ret != 0.0)
    exposure = np.mean(daily_ret != 0.0)
    return {"CAGR": float(ann_ret), "ann_vol": float(ann_vol), "Sharpe": float(sharpe),
            "maxDD": float(mdd), "Calmar": float(calmar),
            "exposure_days_ratio": float(exposure), "active_days": int(turnover), "n_days": int(len(daily_ret))}


# ==================== HYPERPARAM TUNING ====================
def lightweight_random_search(train_ds, val_ds, device, model_type="cnn",
                              n_trials=10, epochs=15, lr_choices=(1e-3,3e-4,1e-4),
                              dropout_choices=(0.1,0.2,0.3), batch_choices=(64,128,256)):
    """
    轻量随机搜索（不依赖 skorch），在单一 train/val split 上挑选超参。
    搜索：学习率、dropout、batch_size；模型：cnn 或 lstm
    """
    best = None; history = []
    for i in range(n_trials):
        lr = random.choice(lr_choices)
        drop = random.choice(dropout_choices)
        bs = random.choice(batch_choices)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate_fn)

        in_channels = train_ds[0][0].shape[0]
        if model_type == "cnn":
            model = TrendCNN1D(in_channels=in_channels, dropout=drop).to(device)
        else:
            model = TrendLSTM(in_channels=in_channels, hidden_size=128, num_layers=2, dropout=drop).to(device)

        criterion = FocalLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val = 0.0
        for ep in range(epochs):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device, criterion)
            va_loss, va_acc, _, _ = evaluate(model, val_loader, device, criterion)
            scheduler.step(va_loss)
            if va_acc > best_val: best_val = va_acc

        history.append((best_val, lr, drop, bs))
        logger.info(f"[Trial {i+1}/{n_trials}] model={model_type} best_val_acc={best_val:.4f} lr={lr} drop={drop} bs={bs}")
        if (best is None) or (best_val > best[0]): best = (best_val, lr, drop, bs)

    logger.info(f"Random search best ({model_type}): val_acc={best[0]:.4f} lr={best[1]} drop={best[2]} bs={best[3]}")
    return {"best_val_acc": best[0], "lr": best[1], "dropout": best[2], "batch_size": best[3], "trials": history}


# ==================== PLOTTING ====================
def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path="outputs/learning_curves.png"):
    plt.figure(figsize=(10,5))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.title("Loss"); plt.xlabel("epoch"); plt.legend()
    # Acc
    plt.subplot(1,2,2)
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.title("Accuracy"); plt.xlabel("epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    logger.info(f"Saved curves to {out_path}")


# ==================== WALK-FORWARD ====================
def generate_walk_forward_splits(
    start_year=2015, end_year=2025, train_years=3, val_months=6,
    embargo_days=None, H=DEFAULT_PREDICTION_HORIZON
):
    warmup_bars = get_warmup_bars()
    if embargo_days is None: embargo_days = H + warmup_bars
    periods = []; cur_train_start = pd.Timestamp(f"{start_year}-01-01")
    while True:
        train_end = cur_train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        val_start = train_end + pd.DateOffset(days=1 + embargo_days)
        val_end   = val_start + pd.DateOffset(months=val_months) - pd.DateOffset(days=1)
        if val_end > pd.Timestamp(f"{end_year}-12-31"): break
        periods.append((cur_train_start.strftime("%Y-%m-%d"),
                        train_end.strftime("%Y-%m-%d"),
                        val_start.strftime("%Y-%m-%d"),
                        val_end.strftime("%Y-%m-%d")))
        cur_train_start = cur_train_start + pd.DateOffset(months=val_months)
    return periods


# ==================== MAIN ====================
def main():
    set_seed(SEED)

    config = {
        'data_path': "E:\\Vesper\\data_downloader\\data\\raw\\HS300",
        'window': DEFAULT_WINDOW_SIZE,
        'H': DEFAULT_PREDICTION_HORIZON,
        'tau': DEFAULT_RETURN_THRESHOLD,
        'batch_size': DEFAULT_BATCH_SIZE,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'weight_decay': DEFAULT_WEIGHT_DECAY,
        'epochs': DEFAULT_EPOCHS,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'num_workers': 0,
        'file_glob': "*.csv",
        'use_focal_loss': True,
        'dropout': DEFAULT_DROPOUT,
        # index / sector
        'mkt_index_path': None,
        'sector_index_dir': None,
        'sector_map_csv': None,
        # backtest
        'proba_threshold': 0.55,
        'fee_rate': 0.001,
        # model choice: "cnn" or "lstm"
        'model_type': "cnn",
        # tuning
        'do_random_search': True,
        'random_search_trials': 8
    }
    logger.info(f"Config: {config}")

    # Build datasets
    try:
        train_ds = Kline1DDataset(
            config['data_path'], window=config['window'], H=config['H'], tau=config['tau'],
            split=("2015-01-01", "2024-06-30"),
            normalize=True, shared_scaler=None,
            mkt_index_path=config['mkt_index_path'],
            sector_index_dir=config['sector_index_dir'],
            sector_map_csv=config['sector_map_csv']
        )
        val_ds = Kline1DDataset(
            config['data_path'], window=config['window'], H=config['H'], tau=config['tau'],
            split=("2024-07-01", "2025-12-31"),
            normalize=True, shared_scaler=train_ds.get_scaler(),
            mkt_index_path=config['mkt_index_path'],
            sector_index_dir=config['sector_index_dir'],
            sector_map_csv=config['sector_map_csv']
        )
    except Exception as e:
        logger.exception(f"Dataset building failed: {e}")
        return

    # Optional: random search (quick)
    tune_result = None
    if config['do_random_search']:
        tune_result = lightweight_random_search(
            train_ds, val_ds, device=config['device'],
            model_type=config['model_type'],
            n_trials=config['random_search_trials'],
            epochs=max(10, config['epochs']//2)
        )
        # apply best params
        config['learning_rate'] = tune_result['lr']
        config['dropout'] = tune_result['dropout']
        config['batch_size'] = tune_result['batch_size']

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], collate_fn=collate_fn,
                              pin_memory=(config['device']=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], collate_fn=collate_fn,
                            pin_memory=(config['device']=="cuda"))

    # Model
    in_channels = train_ds[0][0].shape[0]
    if config['model_type'] == "cnn":
        model = TrendCNN1D(in_channels=in_channels, dropout=config['dropout']).to(config['device'])
    else:
        model = TrendLSTM(in_channels=in_channels, hidden_size=128, num_layers=2,
                          dropout=config['dropout']).to(config['device'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{config['model_type'].upper()} model created with {total_params} params ({trainable_params} trainable)")

    # Loss
    if config['use_focal_loss']:
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    else:
        class_weights = calculate_class_weights(train_ds).to(config['device'])
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optim & sched
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR,
                                                     patience=LR_SCHEDULER_PATIENCE)

    # Training loop + curves
    best_val_acc = 0.0; patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    logger.info("Start training")

    try:
        for epoch in range(config['epochs']):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, config['device'], criterion)
            va_loss, va_acc, val_preds, val_labels = evaluate(model, val_loader, config['device'], criterion)

            scheduler.step(va_loss)
            train_losses.append(tr_loss); val_losses.append(va_loss)
            train_accs.append(tr_acc);   val_accs.append(va_acc)

            logger.info(f"Epoch {epoch+1:03d} | Train {tr_loss:.4f}/{tr_acc:.4f} | "
                        f"Val {va_loss:.4f}/{va_acc:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

            if va_acc > best_val_acc:
                best_val_acc = va_acc; patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': train_ds.get_scaler(),
                    'config': config, 'epoch': epoch, 'val_acc': va_acc,
                    'feature_columns': get_feature_columns()
                }, 'best_model.pth')
                logger.info(f"New best val acc: {best_val_acc:.4f} (model saved)")
            else:
                patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return

    # Curves
    try:
        plot_curves(train_losses, val_losses, train_accs, val_accs, out_path="outputs/learning_curves.png")
    except Exception as e:
        logger.warning(f"Plot curves failed: {e}")

    # Final classification report
    try:
        logger.info("\n" + "="*50 + "\nFINAL VALIDATION RESULTS\n" + "="*50)
        print("Classification Report:")
        print(classification_report(val_labels, val_preds, digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(val_labels, val_preds))
    except Exception as e:
        logger.warning(f"Final metrics failed: {e}")

    # Probabilities + backtests
    try:
        logger.info("\n" + "="*50 + "\nBACKTEST: Fixed-H & Daily Rebalance\n" + "="*50)
        val_probas = predict_proba(model, val_loader, config['device'], mc_dropout_passes=1)

        val_df_list, val_idx_per_stock = rebuild_indices_for_split(
            config['data_path'], config['file_glob'],
            split=("2024-07-01", "2025-12-31"),
            window=config['window'], H=config['H'],
            scaler=train_ds.get_scaler(),
            mkt_index_path=config['mkt_index_path'],
            sector_index_dir=config['sector_index_dir'],
            sector_map_csv=config['sector_map_csv']
        )
        total_val_samples = sum(len(ix) for ix in val_idx_per_stock)
        assert total_val_samples == len(val_probas), f"indices({total_val_samples}) != probas({len(val_probas)})"

        trade_rs, exposure, turnover = backtest_long_flat_fixed_h(
            val_df_list, val_idx_per_stock, val_probas,
            H=config['H'], threshold=config['proba_threshold'], fee_rate=config['fee_rate']
        )
        bt_summary = summarize_backtest(trade_rs, exposure, turnover, periods_per_year=252, holding_period=config['H'])
        print("Fixed-H backtest summary:")
        for k, v in bt_summary.items():
            print(f" - {k}: {v}")

        dates, daily_ret = backtest_daily_rebalance_timeseries(
            val_df_list, val_idx_per_stock, val_probas,
            threshold=config['proba_threshold'], H=config['H'], fee_rate=config['fee_rate']
        )
        daily_summary = summarize_daily_timeseries(dates, daily_ret, rf=0.0, periods_per_year=252)
        print("\nDaily rebalancing summary:")
        for k, v in daily_summary.items():
            print(f" - {k}: {v}")
    except Exception as e:
        logger.warning(f"Backtest failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
