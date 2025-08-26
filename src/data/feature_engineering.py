# src/data/feature_engineering.py

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Tuple

class EnhancedFeatureEngineer:
    """Enhanced feature engineering with more indicators"""

    def __init__(self, config):
        self.config = config
        self.column_mapping = {}

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering pipeline"""
        df = df.copy()

        logger.info(f"特征工程开始，原始列名: {list(df.columns)}")
        
        self.column_mapping = self._detect_columns(df)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        logger.info(f"检测到的列映射: {self.column_mapping}")
        
        if not all(col in self.column_mapping for col in required_cols):
            missing_cols = [col for col in required_cols if col not in self.column_mapping]
            logger.warning(f"缺少必要的列: {missing_cols}")
            logger.warning(f"可用的列: {list(df.columns)}")
            return pd.DataFrame()

        df = self._add_enhanced_indicators(df)
        
        # MODIFIED: Call the new cleaning method here
        df = self.clean_features(df)
        
        logger.info(f"特征工程完成，最终列名: {list(df.columns)}")

        return df
        
    # NEW: Added a robust cleaning method to handle NaN and infinity
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Robustly cleans the feature dataframe after calculations.
        """
        # Replace infinite values created during division with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Use forward-fill and then backward-fill to propagate values
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Any remaining NaNs (e.g., at the very start of the dataframe) are filled with 0
        df.fillna(0, inplace=True)
        
        # Final check for any nulls
        if df.isnull().sum().sum() > 0:
            logger.error("NaN values still exist after cleaning! This should not happen.")
            
        return df

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping = {}
        columns = {col.lower(): col for col in df.columns}
        
        # 扩展模式匹配，支持更多列名变体
        patterns = {
            'open': ['open', 'o', 'op'],
            'high': ['high', 'h', 'hi'], 
            'low': ['low', 'l', 'lo'],
            'close': ['close', 'c', 'cl', 'price', 'p'],
            'volume': ['volume', 'vol', 'v', 'amount', 'amt']
        }
        
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern in columns:
                    mapping[key] = columns[pattern]
                    logger.info(f"列 '{columns[pattern]}' 映射为 {key}")
                    break
        
        # 如果没有找到任何列，尝试使用前5列作为OHLCV
        if not mapping and len(df.columns) >= 5:
            logger.warning("未找到标准列名，使用前5列作为OHLCV")
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
            if len(numeric_cols) >= 5:
                mapping = {
                    'open': numeric_cols[0],
                    'high': numeric_cols[1], 
                    'low': numeric_cols[2],
                    'close': numeric_cols[3],
                    'volume': numeric_cols[4]
                }
                logger.info(f"使用数值列: {mapping}")
        
        return mapping

    def _add_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators with robust calculations"""
        close = df[self.column_mapping['close']]
        high = df[self.column_mapping['high']]
        low = df[self.column_mapping['low']]
        volume = df[self.column_mapping['volume']]

        # 添加epsilon防止除零
        epsilon = 1e-8
        
        # 收益率计算 - 添加安全检查
        df['returns'] = close.pct_change().fillna(0)
        df['log_returns'] = np.log((close / (close.shift(1) + epsilon)) + epsilon).fillna(0)
        
        # 移动平均线 - 使用更稳健的计算
        for period in [5, 10, 20, 50]:
            # 简单移动平均
            df[f'sma_{period}'] = close.rolling(window=period, min_periods=1).mean()
            
            # 指数移动平均
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False, min_periods=1).mean()
            
            # 价格与移动平均线的比率 - 添加epsilon防止除零
            df[f'price_sma_ratio_{period}'] = close / (df[f'sma_{period}'] + epsilon)
            df[f'price_ema_ratio_{period}'] = close / (df[f'ema_{period}'] + epsilon)

        # RSI计算 - 改进版本
        df['rsi_14'] = self._calculate_robust_rsi(close, 14)

        # 布林带宽度 - 改进计算
        bb_period = 20
        bb_sma = close.rolling(window=bb_period, min_periods=1).mean()
        bb_std = close.rolling(window=bb_period, min_periods=1).std()
        
        # 使用epsilon防止除零，并限制极端值
        df['bb_width'] = np.clip((bb_std * 4) / (bb_sma + epsilon), 0, 10)
        
        # 添加更多稳健的技术指标
        df['atr_14'] = self._calculate_atr(high, low, close, 14)
        df['macd'], df['macd_signal'] = self._calculate_macd(close)
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(high, low, close)
        
        return df

    def _calculate_robust_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """改进的RSI计算，更稳健"""
        delta = prices.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 使用指数移动平均，更稳健
        avg_gain = gain.ewm(span=period, adjust=False, min_periods=1).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=1).mean()
        
        # 添加epsilon防止除零
        epsilon = 1e-8
        rs = avg_gain / (avg_loss + epsilon)
        rsi = 100 - (100 / (1 + rs))
        
        # 限制在合理范围内
        return np.clip(rsi, 0, 100)
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算平均真实波幅 (ATR)"""
        # 真实波幅计算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # 取最大值
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算移动平均
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast, adjust=False, min_periods=1).mean()
        ema_slow = prices.ewm(span=slow, adjust=False, min_periods=1).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
        
        return macd_line, signal_line
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """计算随机指标"""
        # 计算周期内的最高价和最低价
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        
        # 计算%K
        epsilon = 1e-8
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + epsilon)
        
        # 计算%K的移动平均得到%D
        d_percent = k_percent.rolling(window=3, min_periods=1).mean()
        
        return k_percent, d_percent