# financial_time_series_model.py
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from typing import Dict, List, Tuple, Optional

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
DEFAULT_EPOCHS = 50
GRADIENT_CLIP_VALUE = 1.0
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

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


# ==================== UTILS ====================
def get_warmup_bars():
    return max(ROLLING_WINDOWS.values())


def safe_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    # 只前向填充，避免未来信息泄漏；剩余 NaN 置 0
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
            df['date'] = pd.to_datetime(df['date'])
            date_col_found = True
        if not date_col_found:
            raise ValueError(f"No date column found in {path}")
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error load_csv_any_datecol {path}: {e}")
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
        # 若无 secu_code，取文件名为代码
        if 'secu_code' not in ret_cols:
            code = os.path.splitext(os.path.basename(csv_path))[0]
            df['secu_code'] = str(code)
            ret_cols = ['secu_code'] + ret_cols
        return df[ret_cols]
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def calculate_rsi(prices, window=RSI_WINDOW):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + SMALL_VALUE)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators_single_stock(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret1'] = df['close'].pct_change()
    df['ret5'] = df['close'].pct_change(ROLLING_WINDOWS['ma_short'])
    df['ma5'] = df['close'].rolling(ROLLING_WINDOWS['ma_short']).mean()
    df['ma20'] = df['close'].rolling(ROLLING_WINDOWS['ma_medium']).mean()
    df['ma50'] = df['close'].rolling(ROLLING_WINDOWS['ma_long']).mean()
    df['mom10'] = df['close'].pct_change(ROLLING_WINDOWS['momentum'])
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
        df['amplitude_norm'] = df['amplitude'] / 100.0
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
    df[f'{prefix}ret1'] = df['close'].pct_change()
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
        print(f"Market index file invalid: {mkt_index_path}")
        return None
    return add_index_indicators(idx, prefix='mkt_')


def load_sector_index_features(sector_index_dir: Optional[str], sector_code: Optional[str]) -> Optional[pd.DataFrame]:
    if not sector_index_dir or not sector_code:
        return None
    # 约定：行业指数文件名 = {sector_code}.csv，含 date/close
    f = os.path.join(sector_index_dir, f"{sector_code}.csv")
    if not os.path.exists(f):
        return None
    idx = load_csv_any_datecol(f)
    if idx is None or 'close' not in idx.columns:
        return None
    return add_index_indicators(idx, prefix='sec_')


def load_sector_map(sector_map_csv: Optional[str]) -> Dict[str, str]:
    # 返回 dict: secu_code -> sector_code
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
        self.sample_meta = []  # 与 samples 对齐：[(secu_code, stock_name, date_idx, date)]
        self.scaler = shared_scaler

        print(f"Loading data from {folder} with split {split}")
        csv_files = glob.glob(os.path.join(folder, file_glob))
        if not csv_files:
            raise ValueError(f"No CSV files found in {folder} matching {file_glob}")
        print(f"Found {len(csv_files)} CSV files")

        # 指数数据（市场指数共用；行业指数按股票加载）
        mkt_idx_feat = load_market_index_features(mkt_index_path)
        sector_map = load_sector_map(sector_map_csv)
        warmup_bars = get_warmup_bars()

        all_train_features = []
        all_stock_data = []

        base_features, optional_features = get_feature_columns()
        # 扩展：指数/行业特征列
        ext_base_features = extend_feature_columns_with_index(base_features)

        for fp in csv_files:
            df = load_kline(fp)
            if df is None or len(df) < MIN_STOCK_DATA_POINTS:
                continue

            secu_code = str(df['secu_code'].iloc[0]) if 'secu_code' in df.columns else os.path.splitext(os.path.basename(fp))[0]
            sector_code = sector_map.get(secu_code, None)
            sec_idx_feat = load_sector_index_features(sector_index_dir, sector_code)

            df = add_indicators_single_stock(df)
            # 合并指数/行业特征
            df = merge_index_features(df, mkt_idx_feat, sec_idx_feat)

            labels = make_label_single_stock(df, H, tau)
            df['label'] = labels

            # 确定特征列（存在才用）
            feat_cols = [col for col in ext_base_features if col in df.columns]
            feat_cols.extend([col for col in optional_features if col in df.columns])

            # 训练期用来拟合 scaler 的原始行
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

        if not all_train_features:
            raise ValueError("No valid training data found")

        # 拟合 scaler（仅训练期）
        if self.normalize and self.scaler is None:
            print("Fitting scaler on training data only...")
            train_features_combined = np.vstack(all_train_features)
            self.scaler = StandardScaler()
            self.scaler.fit(train_features_combined)
            print(f"Scaler fitted on {train_features_combined.shape[0]} training samples")

        # 生成样本
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

        print(f"Created {valid_samples} valid samples from {len(all_stock_data)} stocks")
        if len(self.samples) == 0:
            raise ValueError("No valid samples created. Check parameters and data quality.")

        pos_ratio = label_counts[1] / sum(label_counts.values()) if sum(label_counts.values()) else 0.0
        print(f"Class distribution: {label_counts}")
        print(f"Positive class ratio: {pos_ratio:.3f}")
        if pos_ratio < SEVERE_IMBALANCE_THRESHOLD_LOW or pos_ratio > SEVERE_IMBALANCE_THRESHOLD_HIGH:
            print(f"⚠️  Severe class imbalance! Consider adjusting tau={tau}")

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


# ==================== MODEL ====================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=5, padding=2, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
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
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out += residual
        out = F.gelu(out)
        out = self.dropout(out)
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class TrendCNN1D(nn.Module):
    def __init__(self, in_channels, n_classes=2, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, HIDDEN_DIMS['conv1'], CONV_KERNEL_SIZES['conv1'],
                               CONV_KERNEL_SIZES['conv1'] // 2, dropout)
        self.conv2 = ConvBlock(HIDDEN_DIMS['conv1'], HIDDEN_DIMS['conv2'], CONV_KERNEL_SIZES['conv2'],
                               CONV_KERNEL_SIZES['conv2'] // 2, dropout)
        self.conv3 = ConvBlock(HIDDEN_DIMS['conv2'], HIDDEN_DIMS['conv3'], CONV_KERNEL_SIZES['conv3'],
                               CONV_KERNEL_SIZES['conv3'] // 2, dropout)
        self.res1 = ResidualBlock(HIDDEN_DIMS['conv2'], CONV_KERNEL_SIZES['res1'], dropout)
        self.res2 = ResidualBlock(HIDDEN_DIMS['conv3'], CONV_KERNEL_SIZES['res2'], dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIMS['conv3'] * 2, HIDDEN_DIMS['fc1']),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIMS['fc1'], HIDDEN_DIMS['fc2']),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIMS['fc2'], 2)
        )

    def forward(self, x):
        x = self.conv1(x); x = F.max_pool1d(x, 2)
        x = self.conv2(x); x = self.res1(x); x = F.max_pool1d(x, 2)
        x = self.conv3(x); x = self.res2(x)
        avg_pool = self.global_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(x)


# ==================== TRAIN / EVAL ====================
def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
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
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
    return total_loss / total_samples, correct / total_samples


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
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
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
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
                acc_probs = 0.0
                for _ in range(mc_dropout_passes):
                    logits = model(X)
                    acc_probs += torch.softmax(logits, dim=1)[:, 1]
                probs = (acc_probs / mc_dropout_passes).detach().cpu().numpy()
                probs_all.append(probs)
        model.eval()
        return np.concatenate(probs_all)


def calculate_class_weights(dataset):
    labels = [sample[1] for sample in dataset.samples]
    unique, counts = np.unique(labels, return_counts=True)
    total = sum(counts)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)


# ==================== BACKTEST: TRADE-BY-TRADE (FIXED H) ====================
def equity_curve_from_trades(trade_rs, initial_capital=1.0):
    if trade_rs.size == 0:
        return np.array([initial_capital])
    return initial_capital * np.cumprod(1.0 + trade_rs)


def max_drawdown(equity):
    peak = -np.inf
    mdd = 0.0
    dd_dur = 0
    max_dd_dur = 0
    for v in equity:
        if v > peak:
            peak = v
            dd_dur = 0
        else:
            dd = 1.0 - v / (peak + SMALL_VALUE)
            mdd = max(mdd, dd)
            dd_dur += 1
            max_dd_dur = max(max_dd_dur, dd_dur)
    return mdd, max_dd_dur


def annualize_return(total_ret, periods_per_year, n_periods):
    if n_periods == 0:
        return 0.0
    return (1.0 + total_ret) ** (periods_per_year / n_periods) - 1.0


def sharpe_ratio(ret_series, periods_per_year, rf=0.0):
    if ret_series.size == 0:
        return 0.0
    mean = np.mean(ret_series) - rf/periods_per_year
    vol  = np.std(ret_series, ddof=1)
    if vol <= 0:
        return 0.0
    return (mean * periods_per_year) / (vol * np.sqrt(periods_per_year))


def profit_stats(trade_rs):
    if trade_rs.size == 0:
        return dict(hit_rate=0.0, avg_win=0.0, avg_loss=0.0, win_loss_ratio=0.0, profit_factor=0.0)
    wins = trade_rs[trade_rs > 0]
    losses = trade_rs[trade_rs <= 0]
    hit_rate = wins.size / trade_rs.size if trade_rs.size else 0.0
    avg_win  = wins.mean() if wins.size else 0.0
    avg_loss = -losses.mean() if losses.size else 0.0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    gross_win  = wins.sum() if wins.size else 0.0
    gross_loss = -losses.sum() if losses.size else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.inf
    return dict(hit_rate=float(hit_rate), avg_win=float(avg_win), avg_loss=float(avg_loss),
                win_loss_ratio=float(win_loss_ratio), profit_factor=float(profit_factor))


def backtest_long_flat_fixed_h(
    df_list: List[pd.DataFrame],
    indices_per_stock: List[List[int]],
    probas: np.ndarray,
    H=DEFAULT_PREDICTION_HORIZON,
    threshold=0.55,
    fee_rate=0.001
):
    assert len(probas) == sum(len(ix) for ix in indices_per_stock), "probas size mismatch"
    trade_rs = []
    exposure_flags = []
    turnover_count = 0

    ptr = 0
    for df, idx_list in zip(df_list, indices_per_stock):
        closes = df['close'].to_numpy()
        for i_end in idx_list:
            p = probas[ptr]; ptr += 1
            if p > threshold and (i_end + H) < len(closes):
                px_in  = closes[i_end]     # 用“信号日收盘”开仓（也可改为 i_end+1，视交易规则）
                px_out = closes[i_end + H]
                ret_gross = (px_out - px_in) / (px_in + SMALL_VALUE)
                ret_net = ret_gross - 2.0 * fee_rate
                trade_rs.append(ret_net)
                exposure_flags.append(1)
                turnover_count += 1
            else:
                exposure_flags.append(0)

    trade_rs = np.array(trade_rs) if trade_rs else np.array([])
    exposure_rate = np.mean(exposure_flags) if exposure_flags else 0.0
    return trade_rs, exposure_rate, turnover_count


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
        "trades": int(n_trades),
        "CAGR": float(ann_ret),
        "ann_vol": float(ann_vol),
        "Sharpe": float(sharpe),
        "maxDD": float(mdd),
        "Calmar": float(calmar),
        "hit_rate": pstats["hit_rate"],
        "avg_win": pstats["avg_win"],
        "avg_loss": pstats["avg_loss"],
        "win_loss_ratio": pstats["win_loss_ratio"],
        "profit_factor": pstats["profit_factor"],
        "exposure": float(exposure_rate),
        "turnover_count": int(turnover_count),
        "maxDD_duration_bars": int(max_dd_dur)
    }


# ==================== BACKTEST: DAILY REBALANCE (PORTFOLIO) ====================
def daily_rebalance_backtest(
    val_dataset: Kline1DDataset,
    val_probas: np.ndarray,
    threshold: float = 0.55,
    H: int = DEFAULT_PREDICTION_HORIZON,
    fee_rate: float = 0.001
):
    """
    日频再平衡（重叠持仓，等权聚合）：
    - 每条样本（secu_code, t）若 prob>threshold，则从 t+1 到 t+H 期间“激活持仓”
    - 每天对所有激活持仓等权持有，组合日收益为所有激活股票日收益均值
    - 单笔费用按往返 2*fee_rate 等分到 H 天扣（简化）
    """
    assert len(val_probas) == len(val_dataset.samples)

    # 1) 先把验证集中涉及的所有股票的 date/close 序列收集起来（避免重复读文件，我们直接从 dataset 的 df 无法拿到；
    #    这里用 meta 里的 stock_name 及 secu_code 聚合，再次读原 CSV 再计算 close & date）
    #    由于 dataset 已经将每只股票做了 reset_index，所以我们现在需要源文件路径。
    #    简化：由 stock_name 推回原路径（假设唯一匹配）
    folder = None
    for path_cand in ['.','..']:
        if os.path.isdir(path_cand):
            folder = path_cand
            break

    # 为了稳健，这里从 config 路径外部传较好，但我们可通过 meta 的 stock_name 推回到 data_path：
    # 实际上，我们在 main() 里会直接重建 df_list & idx_per_stock，类似上一个回测函数。
    # 这里我们直接在本函数里重建（需要 val split、window、H、scaler 与 config，同步 main 中逻辑更繁琐）。
    # 为简化使用，我们要求在 main 调用本函数前，提供“每个样本对应的 (df, idx)”结构：
    # ——为避免复杂参数，这里我们复用一个工具来重建（见下）。


def rebuild_indices_for_split(
    folder: str, file_glob: str, split: Tuple[str, str], window: int, H: int,
    scaler: Optional[StandardScaler],
    mkt_index_path: Optional[str],
    sector_index_dir: Optional[str],
    sector_map_csv: Optional[str]
):
    """为某个 split 重建：每只股票的 df(date, close) + 样本结束索引列表（顺序与 dataset/loader 一致）"""
    csv_files = glob.glob(os.path.join(folder, file_glob))
    warmup_bars = get_warmup_bars()

    df_list = []
    idx_per_stock = []

    base_features, optional_features = get_feature_columns()
    ext_base_features = extend_feature_columns_with_index(base_features)

    mkt_idx_feat = load_market_index_features(mkt_index_path)
    sector_map = load_sector_map(sector_map_csv)

    for fp in csv_files:
        df = load_kline(fp)
        if df is None or len(df) < MIN_STOCK_DATA_POINTS:
            continue
        secu_code = str(df['secu_code'].iloc[0]) if 'secu_code' in df.columns else os.path.splitext(os.path.basename(fp))[0]
        sector_code = sector_map.get(secu_code, None)
        sec_idx_feat = load_sector_index_features(sector_index_dir, sector_code)

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
            if not (pd.Timestamp(split[0]) <= date_end <= pd.Timestamp(split[1])):
                continue
            if pd.isna(df.iloc[t]['label']):
                continue
            this_stock_indices.append(t)

        if this_stock_indices:
            df_list.append(df[['date', 'close']].reset_index(drop=True))
            idx_per_stock.append(this_stock_indices)

    return df_list, idx_per_stock


def backtest_daily_rebalance_timeseries(
    df_list: List[pd.DataFrame],
    idx_per_stock: List[List[int]],
    val_probas: np.ndarray,
    threshold: float = 0.55,
    H: int = DEFAULT_PREDICTION_HORIZON,
    fee_rate: float = 0.001
):
    """
    构建“日频收益序列”的回测：
      - 生成全局交易日集合
      - 对每一日，找出所有“处于激活期”的仓位，等权聚合日收益
      - 每笔往返费用 2*fee_rate，等分摊到 H 天
    返回：dates(np.array[datetime64]), daily_returns(np.array[float])
    """
    # 全局日期集合
    all_dates = sorted(set(pd.concat([df['date'] for df in df_list]).unique()))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    daily_ret = np.zeros(len(all_dates), dtype=np.float64)
    daily_npos = np.zeros(len(all_dates), dtype=np.int32)

    ptr = 0
    for df, idxs in zip(df_list, idx_per_stock):
        closes = df['close'].to_numpy()
        dates  = df['date'].to_numpy()
        for t in idxs:
            p = val_probas[ptr]; ptr += 1
            if p <= threshold:
                continue
            # 激活区间：从 t+1 到 t+H （用日收益 r(d)=close[d+1]/close[d]-1）
            start = t + 1
            end   = min(t + H, len(closes) - 2)  # 需要 d 和 d+1
            if start > end:
                continue
            # 费用摊到 H 天
            fee_per_day = (2.0 * fee_rate) / max(1, H)
            for d in range(start, end + 1):
                r = (closes[d + 1] - closes[d]) / (closes[d] + SMALL_VALUE) - fee_per_day
                di = date_to_idx[pd.Timestamp(dates[d])]
                daily_ret[di] += r
                daily_npos[di] += 1

    # 等权聚合
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
    turnover = np.count_nonzero(daily_ret != 0.0)  # 粗略：有仓位的天数
    exposure = np.mean(daily_ret != 0.0)
    return {
        "CAGR": float(ann_ret),
        "ann_vol": float(ann_vol),
        "Sharpe": float(sharpe),
        "maxDD": float(mdd),
        "Calmar": float(calmar),
        "exposure_days_ratio": float(exposure),
        "active_days": int(turnover),
        "n_days": int(len(daily_ret))
    }


# ==================== WALK-FORWARD SPLITS ====================
def generate_walk_forward_splits(
    start_year=2015, end_year=2025,
    train_years=3, val_months=6,
    embargo_days=None, H=DEFAULT_PREDICTION_HORIZON
):
    warmup_bars = get_warmup_bars()
    if embargo_days is None:
        embargo_days = H + warmup_bars
    periods = []
    cur_train_start = pd.Timestamp(f"{start_year}-01-01")
    while True:
        train_end = cur_train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        val_start = train_end + pd.DateOffset(days=1 + embargo_days)
        val_end = val_start + pd.DateOffset(months=val_months) - pd.DateOffset(days=1)
        if val_end > pd.Timestamp(f"{end_year}-12-31"):
            break
        periods.append((
            cur_train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            val_start.strftime("%Y-%m-%d"),
            val_end.strftime("%Y-%m-%d"),
        ))
        cur_train_start = cur_train_start + pd.DateOffset(months=val_months)
    return periods


# ==================== MAIN ====================
def main():
    # Configuration
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
        'num_workers': 0,  # Windows safe
        'file_glob': "*.csv",
        'use_focal_loss': True,
        'dropout': DEFAULT_DROPOUT,

        # === optional index/sector inputs ===
        'mkt_index_path': None,  # 例如: r"E:\...\indexes\sh000300.csv"
        'sector_index_dir': None, # 例如: r"E:\...\sector_indexes"
        'sector_map_csv': None,   # 例如: r"E:\...\sector_map.csv" (secu_code,sector_code)

        # backtest params
        'proba_threshold': 0.55,
        'fee_rate': 0.001,
    }

    print(f"Using device: {config['device']}")
    print(f"Configuration: {config}")

    try:
        # ==== Train dataset (fit scaler) ====
        print("Creating training dataset and fitting scaler...")
        train_ds = Kline1DDataset(
            config['data_path'],
            window=config['window'],
            H=config['H'],
            tau=config['tau'],
            split=("2015-01-01", "2024-06-30"),
            normalize=True,
            shared_scaler=None,
            mkt_index_path=config['mkt_index_path'],
            sector_index_dir=config['sector_index_dir'],
            sector_map_csv=config['sector_map_csv']
        )

        # ==== Validation dataset (shared scaler) ====
        print("Creating validation dataset with shared scaler...")
        val_ds = Kline1DDataset(
            config['data_path'],
            window=config['window'],
            H=config['H'],
            tau=config['tau'],
            split=("2024-07-01", "2025-12-31"),
            normalize=True,
            shared_scaler=train_ds.get_scaler(),
            mkt_index_path=config['mkt_index_path'],
            sector_index_dir=config['sector_index_dir'],
            sector_map_csv=config['sector_map_csv']
        )

        # DataLoaders
        train_loader = DataLoader(
            train_ds, batch_size=config['batch_size'], shuffle=True,
            num_workers=config['num_workers'], collate_fn=collate_fn,
            pin_memory=True if config['device'] == 'cuda' else False
        )
        val_loader = DataLoader(
            val_ds, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], collate_fn=collate_fn,
            pin_memory=True if config['device'] == 'cuda' else False
        )

        # Model
        model = TrendCNN1D(
            in_channels=train_ds[0][0].shape[0],
            dropout=config['dropout']
        ).to(config['device'])
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created with {total_params} total params ({trainable_params} trainable)")

        # Loss
        if config['use_focal_loss']:
            criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
            print(f"Using Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
        else:
            class_weights = calculate_class_weights(train_ds).to(config['device'])
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted CrossEntropy with weights: {class_weights}")

        # Optim & sched
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
        )

        # Train
        best_val_acc = 0.0
        patience_counter = 0
        print("Starting training...")
        for epoch in range(config['epochs']):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, config['device'], criterion)
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, config['device'], criterion)
            scheduler.step(val_loss)
            print(f"Epoch {epoch+1:3d}: Train {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val {val_loss:.4f}/{val_acc:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': train_ds.get_scaler(),
                    'config': config,
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'feature_columns': get_feature_columns()
                }, 'best_model.pth')
                print(f"✅ New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Training completed. Best val acc: {best_val_acc:.4f}")
        print("\n" + "="*50 + "\nFINAL VALIDATION RESULTS\n" + "="*50)
        print("Classification Report:")
        print(classification_report(val_labels, val_preds, digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(val_labels, val_preds))

        # ==== Probabilities on validation ====
        print("\n" + "="*50)
        print("BACKTEST ON VALIDATION (long/flat, hold H days)")
        print("="*50)
        val_probas = predict_proba(model, val_loader, config['device'], mc_dropout_passes=1)

        # ==== Rebuild indices for validation split ====
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

        # ==== Backtest 1: Trade-by-trade (fixed H) ====
        trade_rs, exposure, turnover = backtest_long_flat_fixed_h(
            val_df_list, val_idx_per_stock, val_probas,
            H=config['H'], threshold=config['proba_threshold'], fee_rate=config['fee_rate']
        )
        bt_summary = summarize_backtest(trade_rs, exposure, turnover, periods_per_year=252, holding_period=config['H'])
        print("Fixed-H backtest summary:")
        for k, v in bt_summary.items():
            print(f" - {k}: {v}")

        # ==== Backtest 2: Daily rebalancing (overlapping holdings) ====
        print("\n" + "="*50)
        print("BACKTEST ON VALIDATION (Daily Rebalance, overlapping holdings)")
        print("="*50)
        dates, daily_ret = backtest_daily_rebalance_timeseries(
            val_df_list, val_idx_per_stock, val_probas,
            threshold=config['proba_threshold'], H=config['H'], fee_rate=config['fee_rate']
        )
        daily_summary = summarize_daily_timeseries(dates, daily_ret, rf=0.0, periods_per_year=252)
        print("Daily rebalancing summary:")
        for k, v in daily_summary.items():
            print(f" - {k}: {v}")

        print("\nDone.")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
