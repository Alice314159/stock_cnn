# src/data/multi_stock_processor.py

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import Dict, List, Tuple, Optional

from .feature_engineering import EnhancedFeatureEngineer
from .labeling import make_binary_labels

class MultiStockProcessor:
    """
    多股票数据处理器
    
    功能:
    1. 读取指定文件夹下的所有股票CSV文件
    2. 为每只股票独立进行特征工程和标签生成
    3. 先分割数据，再为每只股票单独训练Scaler和特征选择器
    4. 整合所有股票数据为统一的训练/验证/测试集
    """
    
    def __init__(self, 
                 data_folder: str = None,
                 stock_file_pattern: str = "*.csv",
                 min_stock_count: int = 5,
                 window_size: int = 60,
                 stride: int = 1,
                 prediction_horizon: int = 5,
                 binary_threshold: float = 0.005,
                 max_features: int = 50,
                 feature_selection: str = "correlation",
                 
                 # 预处理器配置
                 preprocessor_config: Optional[Dict] = None,
                 use_advanced_preprocessing: bool = True,
                 
                 # 向后兼容：支持旧的config参数
                 config: Optional[Dict] = None):
        
        # 如果传入了config，使用config中的值
        if config is not None:
            if hasattr(config, 'data'):
                data_config = config.data
                data_folder = getattr(data_config, 'data_folder', data_folder)
                stock_file_pattern = getattr(data_config, 'stock_file_pattern', stock_file_pattern)
                min_stock_count = getattr(data_config, 'min_stock_count', min_stock_count)
                window_size = getattr(data_config, 'window_size', window_size)
                stride = getattr(data_config, 'stride', stride)
                prediction_horizon = getattr(data_config, 'prediction_horizon', prediction_horizon)
                binary_threshold = getattr(data_config, 'binary_threshold', binary_threshold)
                max_features = getattr(data_config, 'max_features', max_features)
                feature_selection = getattr(data_config, 'feature_selection', feature_selection)
        
        # 确保data_folder有值
        if data_folder is None:
            raise ValueError("data_folder must be provided either directly or through config")
        
        self.data_folder = Path(data_folder)
        self.stock_file_pattern = stock_file_pattern
        self.min_stock_count = min_stock_count
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.binary_threshold = binary_threshold
        self.max_features = max_features
        self.feature_selection = feature_selection
        
        # 创建特征工程器
        self.feature_engineer = EnhancedFeatureEngineer(None)  # 暂时传入None作为config
        
        # 存储每只股票的原始数据（未缩放）
        self.stock_data: Dict[str, pd.DataFrame] = {}
        
        # 存储每只股票的训练好的Scaler和特征选择器
        self.stock_scalers: Dict[str, StandardScaler] = {}
        self.stock_selectors: Dict[str, SelectKBest] = {}
        self.stock_features: Dict[str, List[str]] = {}
        
        # 预处理器配置
        self.use_advanced_preprocessing = use_advanced_preprocessing
        if preprocessor_config is None:
            preprocessor_config = {
                "scaler_type": "robust",  # 对异常值更鲁棒
                "feature_selection_method": feature_selection,
                "max_features": max_features,
                "outlier_detection": True,
                "outlier_threshold": 3.0,
                "fill_missing": "interpolate",
                "min_data_quality": 0.8
            }
        
        self.preprocessor_config = preprocessor_config
        
        # 最终整合的数据
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.stock_indices: Optional[np.ndarray] = None
        
    def load_stock_files(self) -> List[str]:
        """
        加载股票文件列表
        
        Returns:
            股票文件路径列表
        """
        if not self.data_folder.exists():
            raise FileNotFoundError(f"数据文件夹不存在: {self.data_folder}")
        
        # 查找所有匹配的CSV文件
        pattern = self.data_folder / self.stock_file_pattern
        stock_files = glob.glob(str(pattern))
        
        if len(stock_files) < self.min_stock_count:
            raise ValueError(f"股票文件数量不足: 找到{len(stock_files)}个，需要至少{self.min_stock_count}个")
        
        logger.info(f"找到{len(stock_files)}个股票文件")
        return sorted(stock_files)
    
    def process_single_stock(self, file_path: Path) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """处理单个股票文件"""
        stock_code = file_path.stem
        try:
            df = pd.read_csv(file_path)
            
            # 重置索引以确保从0开始
            df = df.reset_index(drop=True)
            
            logger.info(f"处理股票{stock_code}: 原始数据{len(df)}行")
            
            # 特征工程
            df_featured = self.feature_engineer.engineer_features(df)
            
            if df_featured.empty or df_featured.isnull().values.any():
                logger.warning(f"跳过 {stock_code}，特征工程后仍有NaN值")
                return None, None

            # 生成标签
            df_labeled = make_binary_labels(df_featured, horizon=self.prediction_horizon, threshold=self.binary_threshold)
            df_labeled['stock_code'] = stock_code
            
            # 再次确保索引正确
            df_labeled = df_labeled.reset_index(drop=True)
            
            logger.info(f"股票{stock_code}处理完成: {len(df_labeled)}行")
            
            return stock_code, df_labeled
        except Exception as e:
            logger.error(f"处理 {stock_code} 失败: {e}")
            return None, None
    
    def process_all_stocks(self) -> None:
        """处理所有股票数据"""
        
        stock_files = self.load_stock_files()
        
        logger.info("开始处理股票数据...")
        for file_path in tqdm(stock_files, desc="处理股票数据"):
            stock_code, df_processed = self.process_single_stock(Path(file_path))
            if stock_code is not None and df_processed is not None:
                self.stock_data[stock_code] = df_processed
        
        if len(self.stock_data) < self.min_stock_count:
            raise ValueError(f"成功处理的股票数量不足: {len(self.stock_data)}，需要至少{self.min_stock_count}个")
        
        logger.info(f"成功处理 {len(self.stock_data)} 只股票")
    
    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """分割数据为训练、验证、测试集"""
        all_data = []
        stock_indices = []
        
        logger.info("开始数据分割...")
        logger.info(f"股票数量: {len(self.stock_data)}")
        
        for stock_idx, (stock_code, df) in enumerate(self.stock_data.items()):
            logger.info(f"股票{stock_code}: {len(df)}行数据")
            # 确保每个DataFrame的索引都重置为从0开始
            df_reset = df.reset_index(drop=True)
            all_data.append(df_reset)
            stock_indices.extend([stock_idx] * len(df))
        
        logger.info(f"总数据行数: {sum(len(df) for df in all_data)}")
        logger.info(f"股票索引范围: {min(stock_indices)} - {max(stock_indices)}")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data['stock_index'] = stock_indices
        
        # 再次重置索引以确保索引从0开始，避免大数值索引
        combined_data = combined_data.reset_index(drop=True)
        
        logger.info(f"合并后数据形状: {combined_data.shape}")
        logger.info(f"合并后索引范围: {combined_data.index.min()} - {combined_data.index.max()}")
        
        # 调试：检查前几行和后几行的索引
        logger.info(f"前5行索引: {combined_data.head().index.tolist()}")
        logger.info(f"后5行索引: {combined_data.tail().index.tolist()}")
        
        # 检查股票索引分布
        stock_idx_counts = combined_data['stock_index'].value_counts().sort_index()
        logger.info(f"股票索引分布: {stock_idx_counts.to_dict()}")
        
        # 使用更智能的分割策略：确保每个分割都有足够的样本
        total_samples = len(combined_data)
        
        # 计算每个分割的最小样本数（考虑窗口大小）
        min_samples_per_split = max(self.window_size * 2, 100)  # 至少是窗口大小的2倍，且不少于100
        
        # 调整分割比例以确保每个分割都有足够的样本
        if total_samples < min_samples_per_split * 3:
            logger.warning(f"数据总量({total_samples})不足以支持标准分割，使用更保守的分割")
            train_ratio = 0.8
            val_ratio = 0.1
        
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        # 确保每个分割都有足够的样本
        if train_end < min_samples_per_split:
            train_end = min_samples_per_split
        if val_end - train_end < min_samples_per_split:
            val_end = train_end + min_samples_per_split
        if total_samples - val_end < min_samples_per_split:
            val_end = total_samples - min_samples_per_split
        
        self.train_data = combined_data.iloc[:train_end].copy()
        self.val_data = combined_data.iloc[train_end:val_end].copy()
        self.test_data = combined_data.iloc[val_end:].copy()
        
        self.stock_indices = np.array(stock_indices)
        
        logger.info(f"数据分割完成 - 训练: {len(self.train_data)}, 验证: {len(self.val_data)}, 测试: {len(self.test_data)}")
        logger.info(f"每个分割都确保至少有{min_samples_per_split}个样本")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_feature_columns(self) -> List[str]:
        """获取特征列名（排除标签和元数据列）"""
        if self.train_data is None:
            raise ValueError("请先调用 process_all_stocks() 和 split_data()")
        
        # 排除非特征列
        exclude_cols = ['label', 'stock_code', 'stock_index', 'Date', 'date']
        feature_cols = [col for col in self.train_data.columns if col not in exclude_cols]
        
        return feature_cols
    
    def save_processed_data(self, output_dir: str) -> None:
        """保存处理后的数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存整合的数据
        if self.train_data is not None:
            self.train_data.to_csv(output_path / "train_data.csv", index=False)
        if self.val_data is not None:
            self.val_data.to_csv(output_path / "val_data.csv", index=False)
        if self.test_data is not None:
            self.test_data.to_csv(output_path / "test_data.csv", index=False)
        
        # 保存股票信息
        stock_info = {
            'stock_codes': list(self.stock_data.keys()),
            'num_stocks': len(self.stock_data),
            'feature_columns': self.get_feature_columns() if self.train_data is not None else []
        }
        
        import json
        with open(output_path / "stock_info.json", 'w', encoding='utf-8') as f:
            json.dump(stock_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到: {output_path}")