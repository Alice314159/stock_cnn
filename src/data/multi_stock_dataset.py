from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from loguru import logger
import os
import pickle


class MultiStockSequenceDataset(Dataset):
    """
    多股票序列数据集 - 支持延迟加载以节省内存
    
    功能:
    1. 将多股票数据转换为适合CNN-LSTM模型的序列格式
    2. 支持股票索引，用于多股票模型
    3. 支持滑动窗口和步长设置
    4. 支持延迟加载和内存优化
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 feature_columns: List[str],
                 window_size: int,
                 stride: int = 1,
                 stock_id_column: str = "stock_code",
                 stock_idx_column: str = "stock_index",
                 label_column: str = "label",
                 use_lazy_loading: bool = True,
                 cache_dir: Optional[str] = None):
        
        self.data = data
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.stride = stride
        self.stock_id_column = stock_id_column
        self.stock_idx_column = stock_idx_column
        self.label_column = label_column
        self.use_lazy_loading = use_lazy_loading
        self.cache_dir = cache_dir
        
        # 检查必要的列是否存在
        missing_columns = []
        if self.stock_idx_column not in data.columns:
            missing_columns.append(self.stock_idx_column)
        if self.label_column not in data.columns:
            missing_columns.append(self.label_column)
        
        if missing_columns:
            raise ValueError(f"数据中缺少必要的列: {missing_columns}")
        
        # 检查窗口大小是否合适
        if window_size > len(data):
            raise ValueError(f"窗口大小({window_size})不能大于数据长度({len(data)})")
        
        # 调试信息
        logger.info(f"数据集信息:")
        logger.info(f"  总行数: {len(data)}")
        logger.info(f"  列名: {list(data.columns)}")
        logger.info(f"  股票索引范围: {data[self.stock_idx_column].min()} - {data[self.stock_idx_column].max()}")
        logger.info(f"  标签分布: {data[self.label_column].value_counts().to_dict()}")
        logger.info(f"  延迟加载: {'启用' if use_lazy_loading else '禁用'}")
        
        # 构建索引
        self.indices = self._build_indices()
        
        # 根据延迟加载策略选择数据存储方式
        if use_lazy_loading:
            self._setup_lazy_loading()
        else:
            self._setup_eager_loading()
        
        logger.info(f"数据集大小: {len(self.indices)}")
        logger.info(f"特征数量: {len(feature_columns)}")
        logger.info(f"窗口大小: {window_size}")
        logger.info(f"步长: {stride}")
    
    def _build_indices(self) -> List[int]:
        """
        构建有效索引列表
        
        Returns:
            有效索引列表
        """
        indices = []
        
        # 按股票分组，为每只股票单独构建索引
        stock_groups = self.data.groupby(self.stock_idx_column)
        
        logger.info(f"开始构建索引，共{len(stock_groups)}个股票组")
        
        # 获取每只股票在原始DataFrame中的实际行位置
        stock_positions = {}
        current_pos = 0
        
        for stock_idx, stock_data in stock_groups:
            stock_length = len(stock_data)
            stock_positions[stock_idx] = (current_pos, current_pos + stock_length - 1)
            current_pos += stock_length
        
        for stock_idx, stock_data in stock_groups:
            start_pos, end_pos = stock_positions[stock_idx]
            
            logger.info(f"股票{stock_idx}: 位置范围 {start_pos} - {end_pos}, 数据长度 {len(stock_data)}")
            
            # 安全检查：确保有足够的数据构建窗口
            if len(stock_data) < self.window_size:
                logger.warning(f"股票{stock_idx}数据不足({len(stock_data)})，跳过")
                continue
            
            # 为这只股票构建滑动窗口索引
            i = start_pos + self.window_size - 1
            while i <= end_pos:
                # 安全检查：确保索引在有效范围内
                if i < len(self.data):
                    indices.append(i)
                else:
                    logger.warning(f"索引{i}超出数据范围{len(self.data)}，跳过")
                    break
                i += self.stride
            
            logger.info(f"股票{stock_idx}: 生成了{len([idx for idx in indices if start_pos <= idx <= end_pos])}个索引")
        
        if not indices:
            raise ValueError("没有生成有效的索引，请检查数据长度和窗口大小设置")
        
        logger.info(f"总共生成了{len(indices)}个有效索引")
        return sorted(indices)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            (特征序列, 股票索引, 标签)
        """
        if idx >= len(self.indices):
            raise IndexError(f"索引{idx}超出范围，最大索引为{len(self.indices)-1}")
        
        end_idx = self.indices[idx]
        start_idx = end_idx - self.window_size + 1
        
        # 安全检查：确保索引在有效范围内
        if start_idx < 0 or end_idx >= len(self.data):
            raise IndexError(f"计算出的索引范围[{start_idx}, {end_idx}]超出数据范围[0, {len(self.data)-1}]")
        
        # 提取特征序列
        if self.use_lazy_loading:
            sequence = self._extract_features_lazy(start_idx, end_idx)
        else:
            sequence = self.feature_data[start_idx:end_idx + 1]  # (window_size, num_features)
        
        # 转换为CNN输入格式 (C=1, H=window_size, W=num_features)
        sequence = sequence[None, :, :]  # (1, window_size, num_features)
        
        # 获取股票索引和标签
        stock_idx = self.stock_indices[end_idx]
        label = self.labels[end_idx]
        
        return (
            torch.from_numpy(sequence),  # (1, window_size, num_features)
            torch.tensor(stock_idx, dtype=torch.long),  # (1,)
            torch.tensor(label, dtype=torch.long)  # (1,)
        )
    
    def get_stock_distribution(self) -> Dict[int, int]:
        """
        获取股票分布统计
        
        Returns:
            股票索引到样本数量的映射
        """
        distribution = {}
        for idx in self.indices:
            stock_idx = self.stock_indices[idx]
            distribution[stock_idx] = distribution.get(stock_idx, 0) + 1
        return distribution
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        获取特征统计信息
        
        Returns:
            特征统计信息字典
        """
        stats = {}
        if self.use_lazy_loading:
            # 延迟加载模式下，特征数据不在内存中，无法直接计算统计
            logger.warning("延迟加载模式下无法获取特征统计，因为特征数据未加载到内存。")
            return {}
        
        for i, col in enumerate(self.feature_columns):
            values = self.feature_data[:, i]
            stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        return stats

    def _setup_lazy_loading(self):
        """设置延迟加载"""
        # 只存储必要的元数据，不加载特征数据到内存
        self.feature_data = None
        self.stock_indices = self.data[self.stock_idx_column].values.astype(np.int64)
        self.labels = self.data[self.label_column].values.astype(np.int64)
        
        # 创建特征列索引映射
        self.feature_col_indices = {col: i for i, col in enumerate(self.feature_columns)}
        
        logger.info("延迟加载模式：特征数据将在需要时从DataFrame中提取")
    
    def _setup_eager_loading(self):
        """设置即时加载（传统模式）"""
        # 转换为numpy数组以提高性能
        self.feature_data = self.data[self.feature_columns].values.astype(np.float32)
        self.stock_indices = self.data[self.stock_idx_column].values.astype(np.int64)
        self.labels = self.data[self.label_column].values.astype(np.int64)
        
        logger.info("即时加载模式：所有特征数据已加载到内存")
    
    def _extract_features_lazy(self, start_idx: int, end_idx: int) -> np.ndarray:
        """延迟加载特征数据"""
        if self.feature_data is not None:
            # 如果已经加载，直接使用
            return self.feature_data[start_idx:end_idx + 1]
        
        # 从DataFrame中提取特征
        features = []
        for i in range(start_idx, end_idx + 1):
            row_features = []
            for col in self.feature_columns:
                value = self.data.iloc[i][col]
                # 处理NaN和无穷大值
                if pd.isna(value) or np.isinf(value):
                    value = 0.0
                row_features.append(float(value))
            features.append(row_features)
        
        return np.array(features, dtype=np.float32)


class MultiStockBatchSampler:
    """
    简化的多股票批次采样器
    
    使用PyTorch内置优化选项，减少复杂度
    """
    
    def __init__(self, dataset: MultiStockSequenceDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 按股票分组索引
        self.stock_groups = self._group_by_stock()
        
        logger.info(f"批次采样器初始化: {len(self.stock_groups)}个股票组")
        for stock_id, indices in self.stock_groups.items():
            logger.info(f"股票{stock_id}: {len(indices)}个样本")
    
    def _group_by_stock(self) -> Dict[int, List[int]]:
        """按股票分组索引"""
        groups = {}
        for i, idx in enumerate(self.dataset.indices):
            stock_idx = self.dataset.stock_indices[idx]
            if stock_idx not in groups:
                groups[stock_idx] = []
            groups[stock_idx].append(i)
        return groups
    
    def __iter__(self):
        """生成批次索引"""
        if not self.stock_groups:
            return
        
        # 获取所有股票ID
        stock_ids = list(self.stock_groups.keys())
        
        if self.shuffle:
            np.random.shuffle(stock_ids)
        
        # 为每个股票创建索引列表
        stock_indices = {stock_id: self.stock_groups[stock_id].copy() for stock_id in stock_ids}
        
        # 如果shuffle，打乱每个股票内部的索引
        if self.shuffle:
            for indices in stock_indices.values():
                np.random.shuffle(indices)
        
        # 生成批次
        batch = []
        while True:
            # 从每个股票中取一个样本
            for stock_id in stock_ids:
                if stock_indices[stock_id]:
                    batch.append(stock_indices[stock_id].pop(0))
                    
                    # 如果批次满了，返回
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
            
            # 检查是否还有足够的样本
            total_remaining = sum(len(indices) for indices in stock_indices.values())
            if total_remaining < self.batch_size:
                # 返回剩余的样本（如果还有的话）
                if batch:
                    yield batch
                break
    
    def __len__(self):
        """计算总批次数"""
        total_samples = sum(len(indices) for indices in self.stock_groups.values())
        return (total_samples + self.batch_size - 1) // self.batch_size


class MultiStockDataLoader:
    """
    优化的多股票数据加载器
    
    使用PyTorch内置优化选项
    """
    
    def __init__(self, 
                 dataset: MultiStockSequenceDataset,
                 batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 prefetch_factor: int = 2,
                 non_blocking: bool = True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.non_blocking = non_blocking
        
        # 创建批次采样器
        self.batch_sampler = MultiStockBatchSampler(dataset, batch_size, shuffle)
        
        # 创建DataLoader，使用PyTorch优化选项
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=self.batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        logger.info(f"数据加载器初始化: workers={num_workers}, persistent={persistent_workers}, prefetch={prefetch_factor}")
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.batch_sampler)
    
    def get_stock_distribution(self) -> Dict[int, int]:
        """获取股票分布"""
        return self.dataset.get_stock_distribution()
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取特征统计"""
        return self.dataset.get_feature_statistics()
