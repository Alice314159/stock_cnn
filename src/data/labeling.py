
import numpy as np
import pandas as pd
from loguru import logger

def make_binary_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.005) -> pd.DataFrame:
    df = df.copy()
    
    # 尝试找到价格列（支持多种可能的列名）
    price_column = None
    possible_price_columns = ['Close', 'close', 'CLOSE', 'price', 'Price', 'PRICE']
    
    for col in possible_price_columns:
        if col in df.columns:
            price_column = col
            break
    
    if price_column is None:
        # 如果找不到价格列，尝试使用第一个数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            price_column = numeric_columns[0]
            logger.warning(f"未找到价格列，使用数值列 '{price_column}' 作为价格")
        else:
            raise ValueError("数据中没有找到价格列或数值列")
    
    logger.info(f"使用列 '{price_column}' 作为价格列")
    
    # Ensure we have enough data for the horizon
    if len(df) <= horizon:
        raise ValueError(f"数据长度({len(df)})不足以支持预测周期({horizon})")
    
    # Calculate future prices and returns
    future = df[price_column].shift(-horizon)
    fwd_ret = (future / df[price_column] - 1.0).astype(np.float32)
    
    # Handle invalid returns
    fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan)
    
    # Create binary labels
    labels = (fwd_ret > threshold).astype(np.int64)  # 1 up, 0 down/flat
    
    # Remove tail with unknown future
    df = df.iloc[:-horizon].copy()
    labels = labels.iloc[:-horizon]
    
    # Ensure labels are valid
    labels = labels.fillna(0).astype(np.int64)
    
    df["label"] = labels.values
    
    # Verify we have valid labels
    if df["label"].isna().any():
        logger.warning("发现无效标签，使用0填充")
        df["label"] = df["label"].fillna(0)
    
    logger.info(f"标签生成完成: 上涨比例 {labels.mean():.2%}")
    
    return df
