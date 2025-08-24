# src/data/feature_engineering.py

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict

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
        """Add comprehensive technical indicators"""
        close = df[self.column_mapping['close']]
        high = df[self.column_mapping['high']]
        low = df[self.column_mapping['low']]
        volume = df[self.column_mapping['volume']]

        df['returns'] = close.pct_change()
        df['log_returns'] = np.log(close / close.shift(1))
        
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = close.rolling(window=period).mean()
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            df[f'price_sma_ratio_{period}'] = close / df[f'sma_{period}']
            df[f'price_ema_ratio_{period}'] = close / df[f'ema_{period}']

        df['rsi_14'] = self._calculate_rsi(close, 14)

        bb_period = 20
        bb_sma = close.rolling(window=bb_period).mean()
        bb_std = close.rolling(window=bb_period).std()
        df['bb_width'] = (bb_std * 4) / (bb_sma + 1e-8)
        
        return df

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8) # Added epsilon to prevent division by zero
        return 100 - (100 / (1 + rs))