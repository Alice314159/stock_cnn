# src/constants/columns.py
from typing import Final

# ── 核心标识列 ───────────────────────────────────────────────────────────
COL_DATE: Final[str]      = "Date"
COL_STOCK_ID: Final[str]  = "stock_id"
COL_LABEL: Final[str]     = "label"

# ── 原始行情列（OHLCV）───────────────────────────────────────────────────
COL_OPEN: Final[str]   = "Open"
COL_HIGH: Final[str]   = "High"
COL_LOW: Final[str]    = "Low"
COL_CLOSE: Final[str]  = "Close"
COL_VOLUME: Final[str] = "Volume"

OHLC_COLS: Final[tuple[str, ...]]  = (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE)
OHLCV_COLS: Final[tuple[str, ...]] = (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME)
REQUIRED_BASE_COLS: Final[frozenset[str]] = frozenset(OHLCV_COLS)

# ── 派生/特征列名 ───────────────────────────────────────────────────────
COL_LOG_CLOSE: Final[str]    = "log_close"
COL_LOG_RETURNS: Final[str]  = "log_returns"

# 命名模板（用于统一滚动/动量/波动率指标列名）
SMA_NAME: Final[str] = "sma_{p}"
EMA_NAME: Final[str] = "ema_{p}"
ROC_NAME: Final[str] = "roc_{p}"
VOL_NAME: Final[str] = "volatility_{p}"

# 其他技术指标
COL_RSI14: Final[str]        = "rsi_14"
COL_MACD: Final[str]         = "macd"
COL_MACD_SIGNAL: Final[str]  = "macd_signal"

# ── 默认参数 ───────────────────────────────────────────────────────────
ROLLING_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20)
RSI_PERIOD: Final[int] = 14
MACD_FAST: Final[int] = 12
MACD_SLOW: Final[int] = 26
MACD_SIGNAL: Final[int] = 9

# ── DType/保留列 ────────────────────────────────────────────────────────
FLOAT_DTYPE: Final[str] = "float32"
NON_FLOAT_COLS: Final[frozenset[str]] = frozenset({COL_DATE})

# ── 可选：别名映射（便于统一外部数据列名）──────────────────────────────
COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    COL_STOCK_ID: ("StockCode", "SecuCode", "Secu_Code", "ticker", "symbol"),
    COL_VOLUME:   ("TurnoverVolume", "volume"),
    COL_CLOSE:    ("Adj Close", "ClosePrice", "close"),
    COL_OPEN:     ("open",),
    COL_HIGH:     ("high",),
    COL_LOW:      ("low",),
}
