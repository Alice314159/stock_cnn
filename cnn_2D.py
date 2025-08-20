# -*- coding: utf-8 -*-
# enhanced_inception_stock_cnn.py
"""
Enhanced Inception Stock CNN with:
1. Inception多尺度架构 - 并行捕获不同时间尺度模式
2. Volume-based features - 关键的成交量特征
3. Adaptive threshold labeling - 自适应阈值标签
4. Loguru日志系统 - 更好的日志输出
5. 全面的数据质量分析
"""

import random, warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from loguru import logger
import sys
import os

warnings.filterwarnings("ignore")


# =========================
# LOGURU 配置
# =========================
def setup_logger():
    """配置loguru日志系统"""
    # 移除默认处理器
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # 添加文件输出
    log_file = "stock_cnn_training.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )

    logger.info("🚀 Logger initialized successfully")
    return log_file


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"🎲 Random seed set to {seed}")


def device_auto():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")
    return device


# =========================
# DATA LOADING (保持原有功能)
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
        logger.error(f"❌ Missing required columns: {miss}")
        raise ValueError(f"Missing required columns: {miss}")

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
    logger.info(f"📊 Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, dtype=str)
        df = adapt_columns(df)
        if asset_code is not None and "asset" in df.columns:
            df = df[df["asset"].astype(str) == str(asset_code)].copy()
        logger.success(f"✅ Successfully loaded {len(df)} rows with columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise


# =========================
# INCEPTION MODULE ARCHITECTURE
# =========================
class StockInceptionModule(nn.Module):
    """
    专门为股票数据设计的Inception模块
    并行使用不同尺寸的卷积核捕获多时间尺度模式
    """

    def __init__(self, in_channels: int, reduce_channels: int = 16, dropout: float = 0.1):
        super().__init__()

        # Branch 1: 1x1卷积 - 捕获当前时点特征
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, kernel_size=1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU()
        )

        # Branch 2: 1x1降维 + 3x1卷积 - 3天模式
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, kernel_size=1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(),
            nn.Conv2d(reduce_channels, reduce_channels * 2, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(reduce_channels * 2),
            nn.ReLU()
        )

        # Branch 3: 1x1降维 + 5x1卷积 - 5天模式
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, kernel_size=1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(),
            nn.Conv2d(reduce_channels, reduce_channels * 2, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(reduce_channels * 2),
            nn.ReLU()
        )

        # Branch 4: 1x1降维 + 7x1卷积 - 7天模式
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, kernel_size=1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(),
            nn.Conv2d(reduce_channels, reduce_channels * 2, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(reduce_channels * 2),
            nn.ReLU()
        )

        # Branch 5: MaxPooling + 1x1卷积 - 保留重要特征
        self.branch5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, reduce_channels, kernel_size=1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU()
        )

        # 特征注意力机制
        total_channels = reduce_channels + reduce_channels * 2 * 3 + reduce_channels  # 1+2+2+2+1倍
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(total_channels // 4, total_channels, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # 并行计算不同分支
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        branch5_out = self.branch5(x)

        # 连接所有分支输出
        concatenated = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out, branch5_out], dim=1)

        # 应用通道注意力
        attention_weights = self.channel_attention(concatenated)
        attended_features = concatenated * attention_weights

        # Dropout正则化
        output = self.dropout(attended_features)

        return output


class AdvancedStockInceptionModule(nn.Module):
    """
    增强版Inception模块，使用膨胀卷积扩大感受野
    能够捕获更长期的依赖关系（10天、15天、20天模式）
    """

    def __init__(self, in_channels: int, reduce_channels: int = 16, dropout: float = 0.1):
        super().__init__()

        # Branch 1: 短期模式 (1-3天)
        self.short_term = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, kernel_size=1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(),
            nn.Conv2d(reduce_channels, reduce_channels * 2, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(reduce_channels * 2),
            nn.ReLU()
        )

        # Branch 2: 中期模式 (5-7天) - 使用膨胀卷积
        self.medium_term = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, kernel_size=1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(),
            nn.Conv2d(reduce_channels, reduce_channels * 2, kernel_size=(3, 1),
                      dilation=(2, 1), padding=(2, 0)),  # 膨胀率=2，等效5天感受野
            nn.BatchNorm2d(reduce_channels * 2),
            nn.ReLU()
        )

        # Branch 3: 长期模式 (10-15天)
        self.long_term = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, kernel_size=1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(),
            nn.Conv2d(reduce_channels, reduce_channels * 2, kernel_size=(3, 1),
                      dilation=(4, 1), padding=(4, 0)),  # 膨胀率=4，等效9天感受野
            nn.BatchNorm2d(reduce_channels * 2),
            nn.ReLU()
        )

        # Branch 4: 超长期模式 (20天+)
        self.extra_long_term = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, kernel_size=1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(),
            nn.Conv2d(reduce_channels, reduce_channels * 2, kernel_size=(3, 1),
                      dilation=(8, 1), padding=(8, 0)),  # 膨胀率=8，等效17天感受野
            nn.BatchNorm2d(reduce_channels * 2),
            nn.ReLU()
        )

        # 跨分支特征交互
        total_channels = reduce_channels * 2 * 4
        self.cross_branch_conv = nn.Sequential(
            nn.Conv2d(total_channels, reduce_channels * 4, kernel_size=1),
            nn.BatchNorm2d(reduce_channels * 4),
            nn.ReLU()
        )

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # 计算各分支输出
        short = self.short_term(x)
        medium = self.medium_term(x)
        long = self.long_term(x)
        extra_long = self.extra_long_term(x)

        # 连接所有分支
        concatenated = torch.cat([short, medium, long, extra_long], dim=1)

        # 跨分支特征交互
        output = self.cross_branch_conv(concatenated)
        output = self.dropout(output)

        return output


# =========================
# ENHANCED INCEPTION CNN WITH VOLUME FEATURES
# =========================
class InceptionStockCNN(nn.Module):
    """
    完整的基于Inception的股票预测CNN
    多层Inception模块 + 渐进式特征提取 + 置信度估计
    """

    def __init__(self,
                 input_features: int,
                 n_classes: int = 2,
                 inception_layers: int = 3,
                 base_channels: int = 32,
                 dropout: float = 0.2):
        super().__init__()

        logger.info(f"🏗️  Building Inception CNN: {inception_layers} layers, {base_channels} base channels")

        # 初始特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout2d(dropout * 0.5)
        )

        # 多层Inception模块
        self.inception_layers = nn.ModuleList()
        current_channels = base_channels

        for i in range(inception_layers):
            if i < inception_layers - 1:
                # 前面的层使用基础Inception
                inception = StockInceptionModule(
                    current_channels,
                    reduce_channels=base_channels // 2,
                    dropout=dropout
                )
                # 计算输出通道数
                current_channels = base_channels // 2 + (base_channels // 2) * 2 * 3 + base_channels // 2
                logger.debug(f"📊 Inception layer {i + 1}: {current_channels} output channels")
            else:
                # 最后一层使用增强Inception
                inception = AdvancedStockInceptionModule(
                    current_channels,
                    reduce_channels=base_channels // 2,
                    dropout=dropout
                )
                current_channels = (base_channels // 2) * 4
                logger.debug(f"📊 Advanced Inception layer {i + 1}: {current_channels} output channels")

            self.inception_layers.append(inception)

        # 全局特征聚合
        self.global_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        # 置信度估计头
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 初始化权重
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🔧 Model created with {total_params:,} parameters")

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始卷积
        x = self.initial_conv(x)

        # 逐层通过Inception模块
        for i, inception in enumerate(self.inception_layers):
            x = inception(x)

        # 全局特征提取
        global_features = self.global_features(x)

        # 分类预测
        logits = self.classifier(global_features)

        # 置信度评估
        confidence = self.confidence_head(global_features)

        return logits, confidence


# =========================
# ENHANCED VOLUME FEATURE ENGINEERING
# =========================
def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    关键的成交量特征工程 - 这些特征对短期预测至关重要
    """
    logger.info("🔧 Computing volume-based features...")

    # 成交量移动平均
    df["volume_sma_5"] = df["volume"].rolling(5).mean()
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_sma_50"] = df["volume"].rolling(50).mean()

    # 成交量比率 (相对成交量强度)
    df["volume_ratio_5"] = df["volume"] / (df["volume_sma_5"] + 1e-8)
    df["volume_ratio_20"] = df["volume"] / (df["volume_sma_20"] + 1e-8)
    df["volume_ratio_50"] = df["volume"] / (df["volume_sma_50"] + 1e-8)

    # 成交量趋势
    df["volume_trend"] = df["volume_sma_5"] / (df["volume_sma_20"] + 1e-8)

    # 价量特征 (资金流向近似)
    df["price_volume"] = df["close"] * df["volume"]
    df["pv_sma_5"] = df["price_volume"].rolling(5).mean()
    df["pv_ratio"] = df["price_volume"] / (df["pv_sma_5"] + 1e-8)

    # 成交量加权平均价格 (VWAP) 近似
    df["vwap_approx"] = (df["close"] * df["volume"]).rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-8)
    df["price_vwap_ratio"] = df["close"] / (df["vwap_approx"] + 1e-8)

    # 成交量激增检测
    volume_std = df["volume"].rolling(20).std()
    df["volume_spike"] = (df["volume"] - df["volume_sma_20"]) / (volume_std + 1e-8)

    # 价量背离指标
    df["price_change"] = df["close"].pct_change()
    df["volume_change"] = df["volume"].pct_change()
    df["price_volume_divergence"] = df["price_change"] * df["volume_change"]

    logger.success("✅ Volume features computed successfully")
    return df


def enhanced_build_dataframe(df: pd.DataFrame,
                             target_col: str = "close",
                             prediction_mode: str = "adaptive_classification",
                             base_features: Optional[List[str]] = None,
                             volatility_lookback: int = 20,
                             threshold_multiplier: float = 0.5) -> pd.DataFrame:
    """
    增强的特征工程，结合成交量和自适应标签
    """
    logger.info(f"🔧 Starting enhanced feature engineering with {prediction_mode} mode")

    df = df.copy()
    if "date" in df.columns:
        df = df.sort_values("date").set_index("date")

    # 基础价格特征
    df["returns"] = df[target_col].pct_change()
    df["log_returns"] = np.log(df[target_col] / df[target_col].shift(1))

    # 价格技术指标
    df["sma_5"] = df[target_col].rolling(5).mean()
    df["sma_10"] = df[target_col].rolling(10).mean()
    df["sma_20"] = df[target_col].rolling(20).mean()
    df["ema_12"] = df[target_col].ewm(span=12).mean()
    df["ema_26"] = df[target_col].ewm(span=26).mean()

    # RSI指标
    df["rsi"] = calculate_rsi(df[target_col], window=14)
    df["rsi_21"] = calculate_rsi(df[target_col], window=21)

    # MACD指标
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # 布林带
    bb_middle = df[target_col].rolling(20).mean()
    bb_std = df[target_col].rolling(20).std()
    df["bb_upper"] = bb_middle + (bb_std * 2)
    df["bb_lower"] = bb_middle - (bb_std * 2)
    df["bb_position"] = (df[target_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)

    # 波动率
    df["volatility"] = df["returns"].rolling(10).std()
    df["volatility_20"] = df["returns"].rolling(20).std()

    # 价格动量
    df["roc_5"] = df[target_col].pct_change(5)
    df["roc_10"] = df[target_col].pct_change(10)

    # 日内特征
    df["hl_ratio"] = (df["high"] - df["low"]) / (df[target_col] + 1e-8)
    df["oc_ratio"] = (df["open"] - df[target_col]) / (df[target_col] + 1e-8)

    # **添加关键的成交量特征**
    df = calculate_volume_features(df)

    # 价格比率
    df["price_sma5_ratio"] = df[target_col] / (df["sma_5"] + 1e-8)
    df["price_sma20_ratio"] = df[target_col] / (df["sma_20"] + 1e-8)

    # === 增强的标签策略 ===
    future_return = df[target_col].shift(-1) / df[target_col] - 1

    if prediction_mode == "adaptive_classification":
        logger.info(f"📊 Using adaptive classification with threshold_multiplier={threshold_multiplier}")
        # 使用滚动波动率创建自适应阈值
        rolling_vol = df["returns"].rolling(volatility_lookback).std()
        up_threshold = threshold_multiplier * rolling_vol
        down_threshold = -threshold_multiplier * rolling_vol

        # 创建自适应阈值标签
        labels = pd.Series(index=df.index, dtype=float)
        labels[future_return > up_threshold] = 1  # 强势上涨
        labels[future_return < down_threshold] = 0  # 强势下跌
        # 中间范围保持NaN（过滤掉）

        df["target"] = labels
        valid_samples = labels.notna().sum()
        up_samples = (labels == 1).sum()
        down_samples = (labels == 0).sum()

        logger.info(f"📈 Adaptive labeling results:")
        logger.info(f"   Valid samples: {valid_samples}")
        logger.info(f"   Up samples: {up_samples} ({up_samples / valid_samples * 100:.1f}%)")
        logger.info(f"   Down samples: {down_samples} ({down_samples / valid_samples * 100:.1f}%)")

    elif prediction_mode == "magnitude_regression":
        # 预测收益率幅度（绝对值）
        df["target"] = np.abs(future_return)

    elif prediction_mode == "direction_magnitude":
        # 创建两个目标：方向和幅度
        df["target_direction"] = (future_return > 0).astype(float)
        df["target_magnitude"] = np.abs(future_return)
        df["target"] = df["target_direction"]  # 主要目标

    else:  # simple_binary
        logger.info("📊 Using simple binary classification")
        valid = df[target_col].notna() & df[target_col].shift(-1).notna()
        df["target"] = np.where(valid, (df[target_col].shift(-1) > df[target_col]).astype(float), np.nan)

    # 增强的特征选择
    volume_features = [
        "volume_ratio_5", "volume_ratio_20", "volume_ratio_50",
        "volume_trend", "pv_ratio", "price_vwap_ratio", "volume_spike",
        "price_volume_divergence"
    ]

    price_features = [
        "returns", "log_returns",
        "sma_5", "sma_20", "rsi", "rsi_21",
        "macd", "macd_histogram", "bb_position",
        "volatility", "volatility_20",
        "roc_5", "roc_10", "hl_ratio", "oc_ratio",
        "price_sma5_ratio", "price_sma20_ratio"
    ]

    if base_features is None:
        base_features = price_features + volume_features

    # 添加任何额外的可用特征
    for c in ["amplitude", "turnover_rate", "chg_pct"]:
        if c in df.columns and c not in base_features:
            base_features.append(c)

    # 最终清理
    out = df[base_features + ["target"]].copy()
    out = out[~out["target"].isna()]  # 移除没有标签的行

    # 填充NaN特征
    feat_cols = [c for c in out.columns if c != "target"]
    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 根据模式转换目标
    if prediction_mode in ["adaptive_classification", "simple_binary", "direction_magnitude"]:
        out["target"] = out["target"].astype(int)

    logger.success(f"✅ Feature engineering complete: {len(out)} samples, {len(feat_cols)} features")
    return out


# =========================
# DATA QUALITY ANALYSIS WITH LOGURU
# =========================
def analyze_data_quality(df: pd.DataFrame, target_col: str = "target"):
    """
    全面的数据质量分析
    """
    logger.info("📊 Starting data quality analysis...")

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Features: {len([c for c in df.columns if c != target_col])}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Time span: {(df.index.max() - df.index.min()).days} days")

    # 目标分布
    if target_col in df.columns:
        target_counts = df[target_col].value_counts().sort_index()
        logger.info("🎯 Target distribution:")
        for val, count in target_counts.items():
            pct = count / len(df) * 100
            logger.info(f"   Class {val}: {count} ({pct:.1f}%)")

        if len(target_counts) == 2:
            imbalance = target_counts.max() / target_counts.min()
            logger.info(f"   Imbalance ratio: {imbalance:.2f}")
            if imbalance > 3:
                logger.warning(f"⚠️  High class imbalance detected: {imbalance:.2f}")

    # 缺失值分析
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning("❌ Missing values found:")
        for col in missing[missing > 0].index:
            pct = missing[col] / len(df) * 100
            logger.warning(f"   {col}: {missing[col]} ({pct:.1f}%)")
    else:
        logger.success("✅ No missing values")

    # 特征重要性 (与目标的相关性)
    if target_col in df.columns:
        feature_cols = [c for c in df.columns if c != target_col]
        correlations = []

        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                corr = abs(df[col].corr(df[target_col]))
                if not pd.isna(corr):
                    correlations.append((col, corr))

        correlations.sort(key=lambda x: x[1], reverse=True)
        logger.info("🔝 Top 10 most predictive features:")
        for i, (col, corr) in enumerate(correlations[:10], 1):
            logger.info(f"   {i:2d}. {col:<25}: {corr:.4f}")

    # 数据分布检查
    feature_cols = [c for c in df.columns if c != target_col and c != 'date']

    # 检查潜在问题
    issues = []
    for col in feature_cols:
        zero_pct = (df[col] == 0).sum() / len(df)
        if zero_pct > 0.5:
            issues.append(f"{col}: {zero_pct * 100:.1f}% zeros")
        if df[col].std() < 1e-6:
            issues.append(f"{col}: near-constant values")

    if issues:
        logger.warning("⚠️  Potential issues found:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    else:
        logger.success("✅ No obvious distribution issues")

    logger.success("📊 Data quality analysis completed")


# =========================
# ENHANCED DATASET
# =========================
class HeatmapDataset(Dataset):
    """增强的数据集类，支持数据增强"""

    def __init__(self, X_scaled: np.ndarray, y: np.ndarray, window: int, augment: bool = False):
        self.X, self.y = self._mkseq(X_scaled, y, window)
        self.augment = augment
        logger.info(f"📦 Dataset created: {len(self.y)} sequences, window_size={window}, augment={augment}")

    @staticmethod
    def _mkseq(X_scaled: np.ndarray, y: np.ndarray, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        Xs, Ys = [], []
        T = len(X_scaled)
        for i in range(W, T):
            Xs.append(X_scaled[i - W:i])
            Ys.append(int(y[i - 1]))
        Xs = np.asarray(Xs, dtype=np.float32)[:, None, :, :]  # 添加通道维度
        Ys = np.asarray(Ys, dtype=np.int64)
        return torch.from_numpy(Xs), torch.from_numpy(Ys)

    def _augment_sequence(self, x):
        """时间序列数据增强"""
        if not self.augment or np.random.random() > 0.3:  # 30%概率增强
            return x

        # 添加小幅随机噪声 (0.5% of std)
        noise_std = torch.std(x) * 0.005
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = self._augment_sequence(x)
        return x, self.y[i]


# =========================
# ENHANCED CONFIG
# =========================
@dataclass
class EnhancedInceptionConfig:
    """增强的配置类"""
    # 模型架构
    window_size: int = 30
    inception_layers: int = 3
    base_channels: int = 32
    dropout: float = 0.2

    # 训练参数
    scaler_type: str = "robust"
    batch_size: int = 64
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # 标签策略
    prediction_mode: str = "adaptive_classification"
    volatility_lookback: int = 20
    threshold_multiplier: float = 0.5

    # 其他
    seed: int = 42
    use_class_weights: bool = True
    use_data_augmentation: bool = True
    early_stopping_patience: int = 15


# =========================
# ENHANCED PIPELINE CLASS
# =========================
class EnhancedInceptionStockCNN:
    """增强的Inception股票CNN流水线"""

    def __init__(self, cfg: EnhancedInceptionConfig, features: Optional[List[str]] = None):
        self.cfg = cfg
        self.features = features
        self.model: Optional[nn.Module] = None
        self.scaler = RobustScaler() if cfg.scaler_type == "robust" else MinMaxScaler()
        self.device = device_auto()

        logger.info(f"🏗️  Initialized pipeline with config: {cfg}")

    def prepare_dataframe(self, df: pd.DataFrame, target_col="close") -> pd.DataFrame:
        """准备数据框"""
        return enhanced_build_dataframe(
            df,
            target_col=target_col,
            base_features=self.features,
            prediction_mode=self.cfg.prediction_mode,
            volatility_lookback=self.cfg.volatility_lookback,
            threshold_multiplier=self.cfg.threshold_multiplier
        )

    @staticmethod
    def time_splits(index: pd.DatetimeIndex, train_ratio=0.7, val_ratio=0.15):
        """时间序列分割"""
        n = len(index)
        i_tr_end = int(n * train_ratio)
        i_va_end = int(n * (train_ratio + val_ratio))
        tr = np.arange(n) < i_tr_end
        va = (np.arange(n) >= i_tr_end) & (np.arange(n) < i_va_end)
        te = np.arange(n) >= i_va_end

        logger.info(f"📊 Data splits: Train={tr.sum()}, Val={va.sum()}, Test={te.sum()}")
        return tr, va, te

    def fit(self, df_clean: pd.DataFrame):
        """训练模型"""
        set_seed(self.cfg.seed)
        logger.info("🚀 Starting model training...")

        # 数据质量分析
        analyze_data_quality(df_clean)

        if len(df_clean) == 0:
            logger.error("❌ Empty dataset after preprocessing!")
            raise ValueError("df_clean has 0 rows.")

        feats = [c for c in df_clean.columns if c != "target"]
        y_all = df_clean["target"].astype(int).values
        X_all = df_clean[feats].astype(np.float32).values
        idx = df_clean.index

        logger.info(f"📊 Model input shape: {X_all.shape}")
        logger.info(f"🔧 Features ({len(feats)}): {feats[:10]}{'...' if len(feats) > 10 else ''}")

        u, c = np.unique(y_all, return_counts=True)
        logger.info(f"🎯 Final label distribution: {dict(zip(u.tolist(), c.tolist()))}")

        classes = np.unique(y_all)
        n_classes = int(classes.max()) + 1

        tr_m, va_m, te_m = self.time_splits(idx, self.cfg.train_ratio, self.cfg.val_ratio)

        # 仅在训练数据上拟合缩放器
        logger.info("📏 Fitting scaler on training data...")
        self.scaler.fit(X_all[tr_m])
        X_tr = self.scaler.transform(X_all[tr_m])
        X_va = self.scaler.transform(X_all[va_m])
        X_te = self.scaler.transform(X_all[te_m])

        W = self.cfg.window_size
        ds_tr = HeatmapDataset(X_tr, y_all[tr_m], W, augment=self.cfg.use_data_augmentation)
        ds_va = HeatmapDataset(X_va, y_all[va_m], W, augment=False)
        ds_te = HeatmapDataset(X_te, y_all[te_m], W, augment=False)

        logger.info(f"📦 Dataset sizes after windowing: train={len(ds_tr)}, val={len(ds_va)}, test={len(ds_te)}")

        if len(ds_tr) == 0 or len(ds_va) == 0:
            logger.error("❌ Insufficient data after windowing!")
            raise ValueError("Insufficient data after windowing")

        dl_tr = DataLoader(ds_tr, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=self.cfg.batch_size, shuffle=False)
        dl_te = DataLoader(ds_te, batch_size=self.cfg.batch_size, shuffle=False)

        # 创建Inception模型
        self.model = InceptionStockCNN(
            input_features=len(feats),
            n_classes=n_classes,
            inception_layers=self.cfg.inception_layers,
            base_channels=self.cfg.base_channels,
            dropout=self.cfg.dropout
        ).to(self.device)

        # 优化器和调度器
        optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)

        # 类别权重处理不平衡
        if self.cfg.use_class_weights:
            weights = compute_class_weight('balanced', classes=np.unique(ds_tr.y.numpy()), y=ds_tr.y.numpy())
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            logger.info(f"⚖️  Using class weights: {weights.cpu().numpy()}")
        else:
            weights = None

        # 训练循环
        history = {"train_loss": [], "val_loss": [], "lr": [], "train_acc": [], "val_acc": []}
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(f"🏋️  Starting training for {self.cfg.epochs} epochs...")

        for ep in range(1, self.cfg.epochs + 1):
            # 训练阶段
            self.model.train()
            tr_loss = tr_correct = n_tr = 0

            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()

                logits, confidence = self.model(xb)

                if weights is not None:
                    loss = nn.functional.cross_entropy(logits, yb, weight=weights)
                else:
                    loss = nn.functional.cross_entropy(logits, yb)

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                tr_loss += float(loss) * len(yb)
                tr_correct += (logits.argmax(1) == yb).sum().item()
                n_tr += len(yb)

            tr_loss /= max(1, n_tr)
            tr_acc = tr_correct / max(1, n_tr)

            # 验证阶段
            self.model.eval()
            va_loss = va_correct = n_va = 0

            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits, confidence = self.model(xb)

                    if weights is not None:
                        loss = nn.functional.cross_entropy(logits, yb, weight=weights)
                    else:
                        loss = nn.functional.cross_entropy(logits, yb)

                    va_loss += float(loss) * len(yb)
                    va_correct += (logits.argmax(1) == yb).sum().item()
                    n_va += len(yb)

            va_loss /= max(1, n_va)
            va_acc = va_correct / max(1, n_va)

            scheduler.step(va_loss)
            cur_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            history["lr"].append(cur_lr)

            # 每10轮记录一次详细信息
            if ep % 10 == 0 or ep <= 5:
                logger.info(f"Epoch {ep:03d} | "
                            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
                            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} | "
                            f"lr={cur_lr:.2e}")

            # 早停
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_inception_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.early_stopping_patience:
                logger.info(f"🛑 Early stopping triggered at epoch {ep}")
                break

        # 加载最佳模型进行测试
        if os.path.exists('best_inception_model.pth'):
            self.model.load_state_dict(torch.load('best_inception_model.pth'))
            logger.info("📥 Loaded best model for testing")

        # 测试评估
        logger.info("🧪 Starting final evaluation...")
        y_true, y_pred, confidences = self.predict_loader_with_confidence(dl_te)
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")

        logger.success("=" * 60)
        logger.success("🎉 FINAL TEST RESULTS")
        logger.success("=" * 60)
        logger.success(f"Test Accuracy: {acc:.4f}")
        logger.success(f"Test F1-macro: {f1m:.4f}")
        logger.info(f"Baseline (random): 0.500")
        improvement = (acc - 0.5) * 100
        logger.success(f"Improvement over random: {improvement:+.1f}%")

        # 置信度分析
        avg_confidence = np.mean(confidences)
        logger.info(f"Average prediction confidence: {avg_confidence:.3f}")

        if acc > 0.52:
            logger.success("🎉 Excellent! Above 52% is very good for stock prediction!")
        elif acc > 0.51:
            logger.success("👍 Good! Above 51% shows meaningful signal.")
        elif acc > 0.505:
            logger.info("📈 Slight edge detected. Consider feature engineering.")
        else:
            logger.warning("📊 No clear signal. Try different features or approach.")

        logger.info("📋 Detailed Classification Report:")
        print(classification_report(y_true, y_pred, digits=4))
        logger.info("🔢 Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        # 分析预测模式
        self.analyze_predictions(y_true, y_pred, confidences)

        logger.success("=" * 60)
        return history, (y_true, y_pred, confidences)

    @torch.no_grad()
    def predict_loader_with_confidence(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """带置信度的预测"""
        self.model.eval()
        y_true, y_pred, confidences = [], [], []
        for xb, yb in loader:
            logits, confidence = self.model(xb.to(self.device))
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(yb.numpy().tolist())
            confidences.extend(confidence.cpu().numpy().flatten().tolist())
        return np.array(y_true), np.array(y_pred), np.array(confidences)

    def analyze_predictions(self, y_true, y_pred, confidences):
        """分析预测结果"""
        logger.info("🔍 Analyzing prediction patterns...")

        # 预测分布
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        true_dist = pd.Series(y_true).value_counts().sort_index()

        logger.info("📊 Prediction vs True Distribution:")
        for cls in sorted(set(y_true) | set(y_pred)):
            pred_count = pred_dist.get(cls, 0)
            true_count = true_dist.get(cls, 0)
            logger.info(f"   Class {cls}: Predicted {pred_count}, True {true_count}")

        # 按类别的性能
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

        logger.info("📈 Per-class Performance:")
        for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
            logger.info(f"   Class {i}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}")

        # 置信度分析
        correct_mask = (y_true == y_pred)
        correct_conf = confidences[correct_mask].mean()
        incorrect_conf = confidences[~correct_mask].mean()

        logger.info(f"🎯 Confidence Analysis:")
        logger.info(f"   Correct predictions confidence: {correct_conf:.3f}")
        logger.info(f"   Incorrect predictions confidence: {incorrect_conf:.3f}")
        logger.info(f"   Confidence gap: {correct_conf - incorrect_conf:.3f}")


# =========================
# 增强的绘图功能
# =========================
def plot_enhanced_inception_history(history: dict, save_path: str = "inception_training_history.png"):
    """增强的训练历史可视化"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Loss曲线
        axes[0, 0].plot(history["train_loss"], label="Train Loss", alpha=0.8, linewidth=2)
        axes[0, 0].plot(history["val_loss"], label="Val Loss", alpha=0.8, linewidth=2)
        axes[0, 0].set_title("📉 Loss Curves", fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[0, 1].plot(history["train_acc"], label="Train Acc", alpha=0.8, linewidth=2)
        axes[0, 1].plot(history["val_acc"], label="Val Acc", alpha=0.8, linewidth=2)
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
        axes[0, 1].set_title("📈 Accuracy Curves", fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 学习率
        axes[0, 2].plot(history["lr"], color='orange', linewidth=2)
        axes[0, 2].set_title("🎛️  Learning Rate Schedule", fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Learning Rate")
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)

        # 验证准确率趋势
        val_acc = np.array(history["val_acc"])
        axes[1, 0].plot(val_acc, color='green', linewidth=3, label="Validation Accuracy")
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
        if len(val_acc) > 10:
            z = np.polyfit(range(len(val_acc)), val_acc, 1)
            trend = np.poly1d(z)(range(len(val_acc)))
            axes[1, 0].plot(trend, '--', alpha=0.8, linewidth=2, label=f'Trend ({z[0] * 100:.3f}%/epoch)')
        axes[1, 0].set_title("🎯 Validation Accuracy Trend", fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Loss差异
        if len(history["train_loss"]) == len(history["val_loss"]):
            loss_diff = np.array(history["val_loss"]) - np.array(history["train_loss"])
            axes[1, 1].plot(loss_diff, color='purple', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].set_title("📊 Overfitting Monitor\n(Val Loss - Train Loss)", fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Loss Difference")
            axes[1, 1].grid(True, alpha=0.3)

        # 性能总结
        axes[1, 2].axis('off')
        final_train_acc = history["train_acc"][-1] if history["train_acc"] else 0
        final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0
        max_val_acc = max(history["val_acc"]) if history["val_acc"] else 0
        final_lr = history["lr"][-1] if history["lr"] else 0

        summary_text = f"""
        📊 Training Summary
        ────────────────────
        Final Train Acc: {final_train_acc:.4f}
        Final Val Acc: {final_val_acc:.4f}
        Best Val Acc: {max_val_acc:.4f}
        Final LR: {final_lr:.2e}

        🎯 vs Random Baseline
        ────────────────────
        Improvement: {(max_val_acc - 0.5) * 100:+.1f}%

        🏆 Model Status
        ────────────────────
        {'🎉 Excellent!' if max_val_acc > 0.52 else '👍 Good!' if max_val_acc > 0.51 else '📈 Needs Work'}
        """

        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        plt.tight_layout()

        # 保存图像
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.success(f"✅ Training plots saved to: {save_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not save plot: {e}")

        # 显示图像
        try:
            plt.show()
        except Exception as e:
            logger.warning(f"⚠️  Could not display plot: {e}")
        finally:
            plt.close(fig)

    except Exception as e:
        logger.error(f"❌ Plotting failed: {e}")
        logger.info("📊 Printing text summary instead...")
        print_text_summary(history)


def print_text_summary(history: dict):
    """文本形式的训练总结"""
    logger.info("📊 TRAINING SUMMARY (Text Format)")
    logger.info("=" * 50)
    if history.get("train_loss"):
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    if history.get("train_acc"):
        logger.info(f"Final train acc: {history['train_acc'][-1]:.4f}")
        logger.info(f"Final val acc: {history['val_acc'][-1]:.4f}")
        logger.info(f"Best val acc: {max(history['val_acc']):.4f}")
    if history.get("lr"):
        logger.info(f"Final learning rate: {history['lr'][-1]:.2e}")
    logger.info("=" * 50)


# =========================
# 主执行函数
# =========================
def main():
    """主执行函数"""
    # 设置日志
    log_file = setup_logger()
    logger.info("🚀 Enhanced Inception Stock CNN Started")
    logger.info("=" * 80)

    set_seed(42)

    # 数据文件路径 - 请修改为您的CSV文件路径
    file_path = r"E:\Vesper\data_downloader\data\raw\indexData\HS300_daily_kline.csv"
    asset_code = None  # 如果是多资产CSV，设置资产代码

    logger.info("📊 Loading market data...")
    try:
        df = load_generic_market_csv(file_path, asset_code=asset_code)
        logger.success(f"✅ Loaded {len(df)} rows")
        logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"💰 Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return

    # 快速数据检查
    logger.info("🔍 Data quality check:")
    logger.info(f"   Missing close values: {df['close'].isna().sum()}")
    logger.info(f"   Missing volume values: {df['volume'].isna().sum()}")
    logger.info(f"   Zero volume days: {(df['volume'] == 0).sum()}")

    # 配置测试 - 测试不同的预测模式
    configs_to_test = [
        {
            "name": "🎯 Adaptive Classification + Inception",
            "config": EnhancedInceptionConfig(
                window_size=30,
                inception_layers=3,
                base_channels=32,
                dropout=0.2,
                prediction_mode="adaptive_classification",
                threshold_multiplier=0.5,
                epochs=60,
                lr=3e-4,
                use_class_weights=True,
                use_data_augmentation=True
            )
        },
        {
            "name": "📈 Simple Binary + Inception",
            "config": EnhancedInceptionConfig(
                window_size=25,
                inception_layers=2,
                base_channels=24,
                dropout=0.15,
                prediction_mode="simple_binary",
                epochs=50,
                lr=4e-4,
                use_class_weights=True,
                use_data_augmentation=False
            )
        }
    ]

    results = {}

    for i, config_test in enumerate(configs_to_test):
        logger.info("=" * 80)
        logger.info(f"🧪 Test {i + 1}/{len(configs_to_test)}: {config_test['name']}")
        logger.info("=" * 80)

        try:
            # 初始化流水线
            cfg = config_test["config"]
            pipe = EnhancedInceptionStockCNN(cfg)

            # 特征工程
            logger.info("🔧 Starting feature engineering...")
            df_clean = pipe.prepare_dataframe(df, target_col="close")

            if len(df_clean) < 500:
                logger.warning(f"⚠️  Too few samples ({len(df_clean)}) after preprocessing!")
                continue

            logger.success(f"✅ Preprocessed dataset: {len(df_clean)} samples")

            # 训练模型
            logger.info("🏋️  Starting model training...")
            history, (y_true, y_pred, confidences) = pipe.fit(df_clean)

            # 存储结果
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")

            results[config_test["name"]] = {
                "accuracy": acc,
                "f1": f1,
                "history": history,
                "predictions": (y_true, y_pred, confidences),
                "config": cfg
            }

            # 绘制结果
            logger.info("📊 Generating training plots...")
            plot_filename = f"inception_training_{i + 1}.png"
            try:
                plot_enhanced_inception_history(history, plot_filename)
            except Exception as plot_err:
                logger.warning(f"⚠️  Plotting failed: {plot_err}")
                print_text_summary(history)

        except Exception as e:
            logger.error(f"❌ Error in {config_test['name']}: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            continue

    # 最终比较
    if results:
        logger.success("=" * 80)
        logger.success("🏆 FINAL COMPARISON")
        logger.success("=" * 80)

        best_result = None
        best_score = 0

        for name, result in results.items():
            acc = result['accuracy']
            f1 = result['f1']
            improvement = (acc - 0.5) * 100

            logger.success(f"📊 {name}")
            logger.info(f"   Accuracy: {acc:.4f}")
            logger.info(f"   F1-macro: {f1:.4f}")
            logger.info(f"   vs Random: {improvement:+.1f}%")

            # 计算综合得分
            composite_score = acc * 0.7 + f1 * 0.3
            logger.info(f"   Composite Score: {composite_score:.4f}")

            if composite_score > best_score:
                best_score = composite_score
                best_result = (name, result)

            logger.info("")

        # 最佳模型
        if best_result:
            best_name, best_data = best_result
            logger.success(f"🥇 BEST MODEL: {best_name}")
            logger.success(f"   Accuracy: {best_data['accuracy']:.4f}")
            logger.success(f"   F1-Score: {best_data['f1']:.4f}")
            logger.success(f"   Composite: {best_score:.4f}")

            # 性能评级
            acc = best_data['accuracy']
            if acc > 0.55:
                logger.success("🎉 OUTSTANDING! This is excellent for stock prediction!")
            elif acc > 0.52:
                logger.success("🎉 EXCELLENT! Very good predictive performance!")
            elif acc > 0.51:
                logger.success("👍 GOOD! Clear predictive signal detected!")
            elif acc > 0.505:
                logger.info("📈 MODERATE: Some signal, needs optimization.")
            else:
                logger.warning("📊 WEAK: Consider different features/approach.")

            # 保存最佳模型配置
            logger.info("💾 Best model configuration:")
            best_cfg = best_data['config']
            logger.info(f"   Window size: {best_cfg.window_size}")
            logger.info(f"   Inception layers: {best_cfg.inception_layers}")
            logger.info(f"   Base channels: {best_cfg.base_channels}")
            logger.info(f"   Dropout: {best_cfg.dropout}")
            logger.info(f"   Learning rate: {best_cfg.lr}")
            logger.info(f"   Prediction mode: {best_cfg.prediction_mode}")

    else:
        logger.error("❌ No successful runs completed!")

    logger.success("=" * 80)
    logger.success("✅ ENHANCED INCEPTION STOCK CNN ANALYSIS COMPLETE!")
    logger.success("=" * 80)

    logger.info("🎯 Key Achievements:")
    logger.info("  ✅ Multi-scale Inception architecture")
    logger.info("  ✅ Volume-based feature engineering")
    logger.info("  ✅ Adaptive threshold labeling")
    logger.info("  ✅ Comprehensive logging with loguru")
    logger.info("  ✅ Advanced data quality analysis")
    logger.info("  ✅ Confidence-based predictions")
    logger.info("  ✅ Early stopping & model persistence")

    logger.info(f"📋 Log file saved to: {log_file}")
    logger.success("=" * 80)


if __name__ == "__main__":
    main()