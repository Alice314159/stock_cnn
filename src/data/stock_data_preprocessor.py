import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from .feature_engineering import engineer_features
from .labeling import make_binary_labels


class StockDataPreprocessor:
    """
    股票数据预处理器
    
    功能:
    1. 数据清洗和验证
    2. 特征工程和选择
    3. 数据标准化/归一化
    4. 异常值检测和处理
    5. 数据质量检查
    """
    
    def __init__(self,
                 scaler_type: str = "standard",  # "standard", "robust", "minmax"
                 feature_selection_method: str = "correlation",  # "correlation", "mutual_info", "random_forest", "pca"
                 max_features: int = 50,
                 outlier_detection: bool = True,
                 outlier_threshold: float = 3.0,
                 fill_missing: str = "forward",  # "forward", "backward", "interpolate", "drop"
                 min_data_quality: float = 0.8):
        
        self.scaler_type = scaler_type
        self.feature_selection_method = feature_selection_method
        self.max_features = max_features
        self.outlier_detection = outlier_detection
        self.outlier_threshold = outlier_threshold
        self.fill_missing = fill_missing
        self.min_data_quality = min_data_quality
        
        # 存储预处理器
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.feature_columns = []
        
    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        数据验证和质量检查
        
        Args:
            df: 输入数据框
            
        Returns:
            (清洗后的数据框, 质量报告)
        """
        original_shape = df.shape
        quality_report = {
            "original_shape": original_shape,
            "missing_values": {},
            "duplicates": 0,
            "outliers": {},
            "data_types": {},
            "cleaned_shape": None
        }
        
        # 检查必需列
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")
        
        # 数据类型检查
        for col in required_cols:
            quality_report["data_types"][col] = str(df[col].dtype)
        
        # 缺失值检查
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                quality_report["missing_values"][col] = missing_count
        
        # 重复值检查
        duplicates = df.duplicated().sum()
        quality_report["duplicates"] = duplicates
        if duplicates > 0:
            df = df.drop_duplicates().reset_index(drop=True)
        
        # 异常值检测
        if self.outlier_detection:
            df, outlier_counts = self._detect_and_handle_outliers(df)
            quality_report["outliers"] = outlier_counts
        
        # 缺失值处理
        df = self._handle_missing_values(df)
        
        # 数据质量评分
        total_cells = original_shape[0] * original_shape[1]
        missing_cells = sum(quality_report["missing_values"].values())
        quality_score = 1 - (missing_cells / total_cells)
        
        if quality_score < self.min_data_quality:
            logger.warning(f"数据质量较低: {quality_score:.2f} < {self.min_data_quality}")
        
        quality_report["cleaned_shape"] = df.shape
        quality_report["quality_score"] = quality_score
        
        return df, quality_report
    
    def _detect_and_handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """检测和处理异常值"""
        outlier_counts = {}
        
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
                
                # 使用中位数替换异常值
                if outliers > 0:
                    median_val = df[col].median()
                    df.loc[df[col] < lower_bound, col] = median_val
                    df.loc[df[col] > upper_bound, col] = median_val
        
        return df, outlier_counts
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        if self.fill_missing == "drop":
            df = df.dropna().reset_index(drop=True)
        elif self.fill_missing == "forward":
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif self.fill_missing == "backward":
            df = df.fillna(method='bfill').fillna(method='ffill')
        elif self.fill_missing == "interpolate":
            df = df.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        try:
            df = engineer_features(df)
            logger.info(f"特征工程完成，特征数量: {len(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"特征工程失败: {e}")
            raise
    
    def select_features(self, df: pd.DataFrame, target_col: str = "label") -> Tuple[pd.DataFrame, List[str]]:
        """
        特征选择
        
        Args:
            df: 输入数据框
            target_col: 目标列名
            
        Returns:
            (特征选择后的数据框, 选择的特征列名列表)
        """
        # 获取数值特征列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ["label", "stock_id", "stock_idx", "Date"]]
        
        if len(feature_cols) <= self.max_features:
            logger.info(f"特征数量({len(feature_cols)})已小于等于最大特征数({self.max_features})，跳过特征选择")
            self.feature_columns = feature_cols
            return df, feature_cols
        
        # 准备特征数据
        X = df[feature_cols].values
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        
        try:
            if self.feature_selection_method == "correlation":
                selector = SelectKBest(score_func=f_classif, k=self.max_features)
            elif self.feature_selection_method == "mutual_info":
                selector = SelectKBest(score_func=mutual_info_classif, k=self.max_features)
            elif self.feature_selection_method == "random_forest":
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                selector = SelectFromModel(rf, max_features=self.max_features)
            elif self.feature_selection_method == "pca":
                # PCA降维
                self.pca = PCA(n_components=min(self.max_features, len(feature_cols)))
                X_transformed = self.pca.fit_transform(X)
                
                # 创建新的特征列名
                pca_features = [f"pca_{i}" for i in range(X_transformed.shape[1])]
                df_pca = pd.DataFrame(X_transformed, columns=pca_features, index=df.index)
                
                # 添加非数值列
                non_numeric_cols = [col for col in df.columns if col not in feature_cols]
                for col in non_numeric_cols:
                    df_pca[col] = df[col]
                
                self.feature_columns = pca_features
                logger.info(f"PCA降维完成，从{len(feature_cols)}个特征降至{len(pca_features)}个")
                return df_pca, pca_features
            else:
                raise ValueError(f"不支持的特征选择方法: {self.feature_selection_method}")
            
            # 拟合特征选择器
            selector.fit(X, y)
            
            if hasattr(selector, 'get_support'):
                selected_indices = selector.get_support()
                selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_indices[i]]
            else:
                # SelectFromModel的情况
                selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.estimator_.feature_importances_[i] > 0]
            
            self.feature_selector = selector
            self.feature_columns = selected_features
            
            logger.info(f"特征选择完成: 从{len(feature_cols)}个特征中选择{len(selected_features)}个")
            
            # 返回选择的特征
            selected_cols = ["Date", "stock_id"] + selected_features + ["label"]
            return df[selected_cols], selected_features
            
        except Exception as e:
            logger.warning(f"特征选择失败: {e}，使用前{self.max_features}个特征")
            selected_features = feature_cols[:self.max_features]
            self.feature_columns = selected_features
            return df, selected_features
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        特征缩放
        
        Args:
            df: 输入数据框
            fit: 是否拟合新的scaler
            
        Returns:
            缩放后的数据框
        """
        if not self.feature_columns:
            logger.warning("没有特征列，跳过缩放")
            return df
        
        # 获取需要缩放的特征列
        scale_cols = [col for col in self.feature_columns if col in df.columns]
        if not scale_cols:
            return df
        
        # 创建或使用scaler
        if fit or self.scaler is None:
            if self.scaler_type == "standard":
                self.scaler = StandardScaler()
            elif self.scaler_type == "robust":
                self.scaler = RobustScaler()
            elif self.scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"不支持的缩放类型: {self.scaler_type}")
            
            # 拟合scaler
            self.scaler.fit(df[scale_cols])
            logger.info(f"拟合{self.scaler_type}缩放器")
        
        # 应用缩放
        df_scaled = df.copy()
        df_scaled[scale_cols] = self.scaler.transform(df[scale_cols])
        
        # 转换为float32以节省内存
        for col in scale_cols:
            df_scaled[col] = df_scaled[col].astype(np.float32)
        
        return df_scaled
    
    def process_single_stock(self, df: pd.DataFrame, stock_id: str) -> Tuple[pd.DataFrame, Dict]:
        """
        处理单只股票数据
        
        Args:
            df: 原始数据框
            stock_id: 股票ID
            
        Returns:
            (处理后的数据框, 处理报告)
        """
        logger.info(f"开始处理股票 {stock_id}")
        
        # 添加股票ID列
        df["stock_id"] = stock_id
        
        # 数据验证
        df, quality_report = self.validate_data(df)
        
        # 特征工程
        df = self.engineer_features(df)
        
        # 生成标签
        df = make_binary_labels(df, horizon=5, threshold=0.005)
        
        # 特征选择
        df, selected_features = self.select_features(df)
        
        # 特征缩放
        df = self.scale_features(df, fit=True)
        
        # 最终清理
        df = df.dropna().reset_index(drop=True)
        
        process_report = {
            "stock_id": stock_id,
            "quality_report": quality_report,
            "selected_features": selected_features,
            "final_shape": df.shape,
            "scaler_type": self.scaler_type,
            "feature_selection_method": self.feature_selection_method
        }
        
        logger.info(f"股票 {stock_id} 处理完成，最终形状: {df.shape}")
        return df, process_report
    
    def get_preprocessor_info(self) -> Dict:
        """获取预处理器信息"""
        info = {
            "scaler_type": self.scaler_type,
            "feature_selection_method": self.feature_selection_method,
            "max_features": self.max_features,
            "outlier_detection": self.outlier_detection,
            "outlier_threshold": self.outlier_threshold,
            "fill_missing": self.fill_missing,
            "min_data_quality": self.min_data_quality,
            "feature_columns": self.feature_columns,
            "has_scaler": self.scaler is not None,
            "has_feature_selector": self.feature_selector is not None,
            "has_pca": self.pca is not None
        }
        return info
