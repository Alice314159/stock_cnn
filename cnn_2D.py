# -*- coding: utf-8 -*-
# stock_cnn2d_pytorch.py
"""
用途：
- 读取日线CSV（支持多种表头：trade_date/secu_code/... 或 Date/StockCode/...）
- 清洗与标准化列名/数值（含科学计数法、中文日期、全角符号、隐形字符等）
- 构造“时间×特征”的2D输入张量（[B, 1, window, n_features]）
- 用2D CNN做二分类（默认：明日涨=1/跌=0），可切换三分类（牛/震/熊）
- 时间顺序切分训练/验证/测试；仅在训练集拟合Scaler，避免泄露
"""

import random, warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
# 在 import matplotlib.pyplot as plt 之后立刻加：
import os
# PyCharm 有环境变量 PYCHARM_HOSTED；在其环境里把后端切成 TkAgg（或退化为 Agg）
if os.environ.get("PYCHARM_HOSTED"):
    try:
        # 如果系统有 Tk，可用交互窗口
        plt.switch_backend("TkAgg")
    except Exception:
        # 无图形界面时用无交互后端
        plt.switch_backend("Agg")


# =========================
# Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Schema-aware CSV loader
# =========================
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """把字符串数值（含逗号、科学计数法）转成数值；无法解析变 NaN。"""
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
    """
    将不同数据源的列名统一到：date, asset(可选), open, high, low, close, volume,
    以及可选列：turnover_value, amplitude, turnover_rate, chg_pct, chg_amt
    同时强力清洗 date 字符串并解析为时间戳。
    """
    df = df.copy()

    # 统一列名：小写，移除空格/下划线/括号/%/短横线/全角斜杠
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

    # 归一化映射表（以规范名为值）
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

    # 必须列
    required = ["date","open","high","low","close","volume"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # ---- 强力清洗 date 后解析 ----
    raw_date = (
        df["date"].astype(str)
          .str.replace("\ufeff", "", regex=False)  # BOM
          .str.replace("\u200b", "", regex=False)  # 零宽空格
          .str.replace("\u00a0", "", regex=False)  # 不间断空格
          .str.replace("／", "/", regex=False)     # 全角斜杠 -> 半角
          .str.replace("年", "/", regex=False)
          .str.replace("月", "/", regex=False)
          .str.replace("日", "", regex=False)
          .str.replace(".", "/", regex=False)
          .str.replace("-", "/", regex=False)
          .str.strip()
    )
    # 依次尝试 M/D/Y, Y/M/D, D/M/Y，最后自动推断
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

    # 按日期清洗排序（注意要赋回 df，而不是丢在临时变量里）
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 数值转型
    numeric_cols = ["open","high","low","close","volume",
                    "turnover_value","amplitude","turnover_rate","chg_pct","chg_amt"]
    df = _coerce_numeric(df, [c for c in numeric_cols if c in df.columns])

    # 如果可选列全为空，删除之
    for c in ["turnover_value","amplitude","turnover_rate","chg_pct","chg_amt"]:
        if c in df.columns and df[c].isna().all():
            df.drop(columns=[c], inplace=True)

    return df

def load_generic_market_csv(file_path: str, asset_code: Optional[str] = None) -> pd.DataFrame:
    """
    读取CSV并调用 adapt_columns 统一格式。
    如果包含多标的（有 asset 列），可通过 asset_code 过滤单一标的。
    """
    df = pd.read_csv(file_path, dtype=str)  # 先按字符串读取，避免代码/科学计数被转换
    df = adapt_columns(df)
    if asset_code is not None and "asset" in df.columns:
        df = df[df["asset"].astype(str) == str(asset_code)].copy()
    return df


# =========================
# Feature engineering
# =========================
def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0); loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def build_dataframe(df: pd.DataFrame,
                    target_col: str = "close",
                    base_features: Optional[List[str]] = None,
                    regime3: bool = False,
                    horizon: int = 1,
                    up: float = 0.02,
                    down: float = -0.02) -> pd.DataFrame:
    """
    生成特征与标签：
    - 默认二分类（明日涨/跌），仅在今天与明天 close 都有效时才打标签；
    - 可选三分类（牛/震/熊），按未来 horizon 天收益与 up/down 阈值划分。
    """
    df = df.copy()
    if "date" in df.columns:
        df = df.sort_values("date").set_index("date")

    # 工程特征（会产生部分 NaN）
    df["returns"] = df[target_col].pct_change()
    df["sma_5"] = df[target_col].rolling(5).mean()
    df["sma_20"] = df[target_col].rolling(20).mean()
    df["rsi"] = calculate_rsi(df[target_col], window=14)
    df["volatility"] = df["returns"].rolling(10).std()

    # 标签
    if regime3:
        fwd = df[target_col].shift(-horizon) / df[target_col] - 1.0
        y = pd.Series(index=df.index, dtype="float64")
        y[fwd < down] = 0
        y[(fwd >= down) & (fwd <= up)] = 1
        y[fwd > up] = 2
        df["target"] = y
    else:
        nxt = df[target_col].shift(-1)
        valid = df[target_col].notna() & nxt.notna()
        df["target"] = np.where(valid, (nxt > df[target_col]).astype(float), np.nan)

    # 默认特征
    if base_features is None:
        base_features = [
            "open","high","low","close","volume",
            "sma_5","sma_20","rsi","volatility"
        ]
    # 合并可选原始列
    for c in ["amplitude","turnover_rate","chg_pct","turnover_value","chg_amt"]:
        if c in df.columns and c not in base_features:
            base_features.append(c)

    # 仅丢弃没有标签的行；特征侧 NaN/Inf 填 0
    out = df[base_features + ["target"]].copy()
    out = out[~out["target"].isna()]
    feat_cols = [c for c in out.columns if c != "target"]
    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["target"] = out["target"].astype(int)
    return out


# =========================
# Dataset (window → 2D)
# =========================
class HeatmapDataset(Dataset):
    """把 [T, F] 滑窗成 [N, 1, window, F]，标签对齐到窗口最后一天。"""
    def __init__(self, X_scaled: np.ndarray, y: np.ndarray, window: int):
        self.X, self.y = self._mkseq(X_scaled, y, window)

    @staticmethod
    def _mkseq(X_scaled: np.ndarray, y: np.ndarray, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        Xs, Ys = [], []
        T = len(X_scaled)
        for i in range(W, T):
            Xs.append(X_scaled[i-W:i])
            Ys.append(int(y[i-1]))  # 标签对应窗口最后一天
        Xs = np.asarray(Xs, dtype=np.float32)[:, None, :, :]  # [N, 1, H, W]
        Ys = np.asarray(Ys, dtype=np.int64)
        return torch.from_numpy(Xs), torch.from_numpy(Ys)

    def __len__(self): return self.y.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


# =========================
# Model
# =========================
class CNN2DRegime(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5,3), padding=(2,1)), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(5,3), padding=(2,1), stride=(2,1)), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=(5,3), padding=(2,1)), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # 不受输入 H×W 的影响，输出固定 1×1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.head(self.feature(x))


# =========================
# Training helpers
# =========================
class EarlyStopper:
    def __init__(self, patience: int = 15, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience; self.mode = mode; self.min_delta = min_delta
        self.best = None; self.bad = 0
        self._better = (lambda cur, best: cur < best - self.min_delta) if mode == "min" else (lambda cur, best: cur > best + self.min_delta)

    def step(self, val):
        if self.best is None: self.best = val; return False
        if self._better(val, self.best): self.best = val; self.bad = 0; return False
        self.bad += 1; return self.bad > self.patience

def class_weights_from_labels(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """按频次反比生成类别权重，缓解不平衡。"""
    bc = np.bincount(y, minlength=n_classes).astype(np.float32)
    bc[bc == 0] = 1.0
    w = 1.0 / bc; w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


# =========================
# Pipeline
# =========================
@dataclass
class TrainConfig:
    window_size: int = 20
    scaler_type: str = "robust"   # or "minmax"
    batch_size: int = 128
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    regime3: bool = False         # True -> 3 类牛/震/熊
    horizon: int = 1              # 三分类使用
    up: float = 0.02              # 三分类阈值
    down: float = -0.02

class StockCNN2DPyTorch:
    def __init__(self, cfg: TrainConfig, features: Optional[List[str]] = None):
        self.cfg = cfg; self.features = features; self.model: Optional[nn.Module] = None
        self.scaler = RobustScaler() if cfg.scaler_type == "robust" else MinMaxScaler()

    def prepare_dataframe(self, df: pd.DataFrame, target_col="close") -> pd.DataFrame:
        return build_dataframe(
            df, target_col=target_col, base_features=self.features,
            regime3=self.cfg.regime3, horizon=self.cfg.horizon,
            up=self.cfg.up, down=self.cfg.down
        )

    @staticmethod
    def time_splits(index: pd.DatetimeIndex, train_ratio=0.7, val_ratio=0.15):
        n = len(index)
        i_tr_end = int(n * train_ratio)
        i_va_end = int(n * (train_ratio + val_ratio))
        tr = np.arange(n) < i_tr_end
        va = (np.arange(n) >= i_tr_end) & (np.arange(n) < i_va_end)
        te = np.arange(n) >= i_va_end
        return tr, va, te

    def fit(self, df_clean: pd.DataFrame):
        set_seed(self.cfg.seed)
        if len(df_clean) == 0:
            raise ValueError("df_clean has 0 rows. Check CSV parsing and feature building.")

        feats = [c for c in df_clean.columns if c != "target"]
        y_all = df_clean["target"].astype(int).values
        X_all = df_clean[feats].astype(np.float32).values
        idx = df_clean.index

        if y_all.size == 0:
            raise ValueError("No labels after preprocessing. Possible causes: bad date parse or all labels NaN.")

        # 标签分布（便于检查不平衡）
        u, c = np.unique(y_all, return_counts=True)
        print(f"Label distribution: {dict(zip(u.tolist(), c.tolist()))}")

        classes = np.unique(y_all)
        n_classes = int(classes.max()) + 1

        tr_m, va_m, te_m = self.time_splits(idx, self.cfg.train_ratio, self.cfg.val_ratio)

        # 仅用训练集拟合 scaler
        self.scaler.fit(X_all[tr_m])
        X_tr = self.scaler.transform(X_all[tr_m])
        X_va = self.scaler.transform(X_all[va_m])
        X_te = self.scaler.transform(X_all[te_m])

        W = self.cfg.window_size
        ds_tr = HeatmapDataset(X_tr, y_all[tr_m], W)
        ds_va = HeatmapDataset(X_va, y_all[va_m], W)
        ds_te = HeatmapDataset(X_te, y_all[te_m], W)

        # 窗口化后是否还有样本
        for name, ds in [("train", ds_tr), ("val", ds_va), ("test", ds_te)]:
            if len(ds) == 0:
                raise ValueError(
                    f"No sequences in {name} set. "
                    f"Reduce window_size (now {W}) or adjust train/val/test ratios."
                )

        dl_tr = DataLoader(ds_tr, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=self.cfg.batch_size, shuffle=False)
        dl_te = DataLoader(ds_te, batch_size=self.cfg.batch_size, shuffle=False)

        self.model = CNN2DRegime(n_classes=n_classes)
        dev = device_auto(); self.model.to(dev)
        opt = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # 兼容旧版 ReduceLROnPlateau（有的版本没有 verbose/min_lr）
        try:
            sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
        except TypeError:
            sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)

        weights = class_weights_from_labels(ds_tr.y.numpy(), n_classes).to(dev)

        early = EarlyStopper(patience=15, mode="min", min_delta=1e-4)
        history = {"train_loss": [], "val_loss": [], "lr": []}

        for ep in range(1, self.cfg.epochs + 1):
            # ---- train ----
            self.model.train(); tr_loss = n_tr = 0
            for xb, yb in dl_tr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                logits = self.model(xb)
                loss = nn.functional.cross_entropy(logits, yb, weight=weights)
                loss.backward(); opt.step()
                tr_loss += float(loss) * len(yb); n_tr += len(yb)
            tr_loss /= max(1, n_tr)

            # ---- val ----
            self.model.eval(); va_loss = n_va = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(dev), yb.to(dev)
                    logits = self.model(xb)
                    loss = nn.functional.cross_entropy(logits, yb, weight=weights)
                    va_loss += float(loss) * len(yb); n_va += len(yb)
            va_loss /= max(1, n_va)

            sch.step(va_loss); cur_lr = opt.param_groups[0]["lr"]
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["lr"].append(cur_lr)
            print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} val_loss={va_loss:.4f} lr={cur_lr:.2e}")

            if early.step(va_loss):
                print("Early stopping."); break

        # ---- test ----
        y_true, y_pred = self.predict_loader(dl_te, dev)
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        print("\n=== Test Results ===")
        print(f"Accuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion matrix:"); print(confusion_matrix(y_true, y_pred))
        return history, (y_true, y_pred)

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader, dev) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        y_true, y_pred = [], []
        for xb, yb in loader:
            logits = self.model(xb.to(dev))
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(yb.numpy().tolist())
        return np.array(y_true), np.array(y_pred)

    @torch.no_grad()
    def predict(self, X_windowed: torch.Tensor) -> np.ndarray:
        """对外推理接口：X_windowed 形状 [B, 1, window, n_features]"""
        if self.model is None: raise RuntimeError("Model not trained.")
        dev = device_auto(); self.model.eval().to(dev)
        logits = self.model(X_windowed.to(dev))
        return nn.functional.softmax(logits, dim=1).cpu().numpy()


# =========================
# Plot helper
# =========================
def plot_history(history: dict):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train_loss")
    ax[0].plot(history["val_loss"], label="val_loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(history["lr"], label="lr"); ax[1].legend(); ax[1].set_title("Learning Rate")
    plt.tight_layout(); plt.show()


# =========================
# Main
# =========================
if __name__ == "__main__":
    set_seed(42)

    # ---- 修改为你的CSV路径 ----
    file_path = r"E:\Vesper\data_downloader\data\raw\indexData\HS300_daily_kline.csv"
    # 如果CSV包含多只标的，想只训练其中一只，填 asset_code，例如 "1.0003" 或 "000300.SH"
    asset_code = None

    print("Loading CSV (schema-agnostic)…")
    df = load_generic_market_csv(file_path, asset_code=asset_code)

    # 加载后快速体检（排查 NaT/脏值）
    print("After load:", df.shape, "| columns:", list(df.columns))
    print(df.head(5).to_string(index=False))
    print("close non-null %:", 1 - df["close"].isna().mean())
    print("next-close non-null %:", 1 - df["close"].shift(-1).isna().mean())

    cfg = TrainConfig(
        window_size=20,
        scaler_type="robust",
        batch_size=128,
        epochs=80,
        lr=1e-3,
        weight_decay=1e-4,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        regime3=False,     # 三分类请改 True，并设置 horizon/up/down
        horizon=10,
        up=0.02, down=-0.02
    )

    pipe = StockCNN2DPyTorch(cfg)
    print("Preparing features/labels…")
    df_clean = pipe.prepare_dataframe(df, target_col="close")
    print(f"Rows after cleaning: {len(df_clean)}, features: {len([c for c in df_clean.columns if c!='target'])}")

    if len(df_clean) == 0:
        tmp = build_dataframe(df)  # 调试：看标签是否创建出来
        print("Non-null target count (debug):", tmp["target"].notna().sum() if "target" in tmp else 0)
        raise SystemExit(1)

    print("Training…")
    history, _ = pipe.fit(df_clean)

    print("Plotting…")
    plot_history(history)

    print("Done.")
