import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import random
from loguru import logger

class StockDataAugmentation:
    """
    股票数据增强类
    
    包含多种适合时序金融数据的增强方法
    """
    
    def __init__(self, 
                 noise_std: float = 0.01,
                 time_shift_max: int = 5,
                 magnitude_warp_sigma: float = 0.2,
                 time_warp_sigma: float = 0.2,
                 cutmix_alpha: float = 1.0,
                 mixup_alpha: float = 0.2,
                 jitter_std: float = 0.005,
                 scaling_factor: float = 0.1,
                 rotation_angle: float = 5.0,
                 masking_ratio: float = 0.1,
                 frequency_dropout: float = 0.1):
        
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.time_warp_sigma = time_warp_sigma
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.jitter_std = jitter_std
        self.scaling_factor = scaling_factor
        self.rotation_angle = rotation_angle
        self.masking_ratio = masking_ratio
        self.frequency_dropout = frequency_dropout
    
    def add_noise(self, x: torch.Tensor, std: Optional[float] = None) -> torch.Tensor:
        """添加高斯噪声"""
        if std is None:
            std = self.noise_std
        noise = torch.randn_like(x) * std
        return x + noise
    
    def time_shift(self, x: torch.Tensor, max_shift: Optional[int] = None) -> torch.Tensor:
        """时间平移"""
        if max_shift is None:
            max_shift = self.time_shift_max
        
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return x
        
        if shift > 0:
            # 右移，前面用第一个值填充
            padded = torch.cat([x[:, :, :1, :].repeat(1, 1, shift, 1), x[:, :, :-shift, :]], dim=2)
        else:
            # 左移，后面用最后一个值填充
            shift = -shift
            padded = torch.cat([x[:, :, shift:, :], x[:, :, -1:, :].repeat(1, 1, shift, 1)], dim=2)
        
        return padded
    
    def magnitude_warp(self, x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """幅度扭曲"""
        if sigma is None:
            sigma = self.magnitude_warp_sigma
        
        # 生成平滑的扭曲因子
        warp_factors = torch.randn(x.size(0), 1, x.size(2), 1) * sigma
        warp_factors = torch.cumsum(warp_factors, dim=2)
        warp_factors = 1 + torch.tanh(warp_factors)  # 确保为正值
        
        return x * warp_factors
    
    def time_warp(self, x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """时间扭曲（简化版）"""
        if sigma is None:
            sigma = self.time_warp_sigma
        
        batch_size, channels, time_steps, features = x.shape
        
        # 生成时间扭曲映射
        warp_amount = int(time_steps * sigma * 0.1)
        if warp_amount == 0:
            return x
        
        warped = x.clone()
        for i in range(batch_size):
            # 随机选择扭曲点
            warp_point = random.randint(warp_amount, time_steps - warp_amount)
            shift = random.randint(-warp_amount, warp_amount)
            
            if shift > 0:
                # 拉伸
                stretched = torch.nn.functional.interpolate(
                    x[i:i+1, :, warp_point-shift:warp_point+shift, :].permute(0, 3, 1, 2),
                    size=(channels, shift * 2),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
                warped[i, :, warp_point-shift:warp_point+shift, :] = stretched[0]
            elif shift < 0:
                # 压缩
                compressed = torch.nn.functional.interpolate(
                    x[i:i+1, :, warp_point+shift:warp_point-shift, :].permute(0, 3, 1, 2),
                    size=(channels, -shift * 2),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
                warped[i, :, warp_point+shift:warp_point-shift, :] = compressed[0]
        
        return warped
    
    def cutmix(self, x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CutMix增强"""
        batch_size, channels, time_steps, features = x1.shape
        
        # 生成lambda
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # 随机选择切割区域
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(features * cut_ratio)
        cut_h = int(time_steps * cut_ratio)
        
        cx = np.random.randint(features)
        cy = np.random.randint(time_steps)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, features)
        bby1 = np.clip(cy - cut_h // 2, 0, time_steps)
        bbx2 = np.clip(cx + cut_w // 2, 0, features)
        bby2 = np.clip(cy + cut_h // 2, 0, time_steps)
        
        # 应用CutMix
        x_mixed = x1.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (features * time_steps))
        
        return x_mixed, lam
    
    def mixup(self, x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MixUp增强"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        return x_mixed, lam
    
    def random_augment(self, x: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
        """随机应用增强方法"""
        if random.random() < prob:
            aug_type = random.choice(['noise', 'time_shift', 'magnitude_warp', 'time_warp'])
            
            if aug_type == 'noise':
                return self.add_noise(x)
            elif aug_type == 'time_shift':
                return self.time_shift(x)
            elif aug_type == 'magnitude_warp':
                return self.magnitude_warp(x)
            elif aug_type == 'time_warp':
                return self.time_warp(x)
        
        return x
    
    def add_jitter(self, x: torch.Tensor, std: Optional[float] = None) -> torch.Tensor:
        """添加抖动（小幅度随机变化）"""
        if std is None:
            std = self.jitter_std
        jitter = torch.randn_like(x) * std
        return x + jitter
    
    def scale_magnitude(self, x: torch.Tensor, factor: Optional[float] = None) -> torch.Tensor:
        """幅度缩放"""
        if factor is None:
            factor = self.scaling_factor
        
        # 随机缩放因子
        scale = 1 + (torch.rand(1) * 2 - 1) * factor
        return x * scale
    
    def random_masking(self, x: torch.Tensor, ratio: Optional[float] = None) -> torch.Tensor:
        """随机掩码（模拟数据缺失）"""
        if ratio is None:
            ratio = self.masking_ratio
        
        mask = torch.rand_like(x) > ratio
        return x * mask
    
    def frequency_dropout(self, x: torch.Tensor, dropout_rate: Optional[float] = None) -> torch.Tensor:
        """频率域dropout（在FFT域中随机丢弃频率）"""
        if dropout_rate is None:
            dropout_rate = self.frequency_dropout
        
        # 转换为频域
        x_fft = torch.fft.rfft(x, dim=2)
        
        # 创建dropout掩码
        mask = torch.rand_like(x_fft.real) > dropout_rate
        
        # 应用dropout
        x_fft_dropped = x_fft * mask
        
        # 转换回时域
        x_dropped = torch.fft.irfft(x_fft_dropped, n=x.size(2), dim=2)
        
        return x_dropped
    
    def apply_augmentation_pipeline(self, x: torch.Tensor, augmentation_prob: float = 0.5) -> torch.Tensor:
        """
        应用完整的增强流水线
        
        Args:
            x: 输入张量
            augmentation_prob: 每个增强方法的应用概率
            
        Returns:
            增强后的张量
        """
        # 检查输入数据是否有效
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("输入数据包含NaN或Inf值，跳过数据增强")
            return x
        
        augmented = x.clone()
        
        # 定义增强方法列表，确保所有方法都是有效的函数
        augmentations = [
            (self.add_noise, {}),
            (self.time_shift, {}),
            (self.magnitude_warp, {}),
            (self.add_jitter, {}),
            (self.scale_magnitude, {}),
            (self.random_masking, {}),
            (self.frequency_dropout, {})
        ]
        
        for aug_func, kwargs in augmentations:
            # 确保aug_func是一个可调用的函数
            if callable(aug_func) and torch.rand(1) < augmentation_prob:
                try:
                    result = aug_func(augmented, **kwargs)
                    # 检查结果是否有效
                    if torch.isnan(result).any() or torch.isinf(result).any():
                        logger.warning(f"增强方法 {aug_func.__name__} 产生无效值，跳过")
                        continue
                    augmented = result
                except Exception as e:
                    logger.warning(f"增强方法 {aug_func.__name__} 失败: {e}")
                    continue
        
        # 最终检查：确保输出数据有效
        if torch.isnan(augmented).any() or torch.isinf(augmented).any():
            logger.warning("数据增强后产生无效值，返回原始数据")
            return x
        
        return augmented


class StockDataPreprocessor:
    """
    股票数据预处理器
    
    包含标准化、特征工程、数据清洗等功能
    """
    
    def __init__(self, 
                 normalization_method: str = 'robust',
                 handle_missing: str = 'interpolate',
                 feature_engineering: bool = True,
                 remove_outliers: bool = True,
                 outlier_threshold: float = 3.0):
        
        self.normalization_method = normalization_method
        self.handle_missing = handle_missing
        self.feature_engineering = feature_engineering
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        
        # 存储标准化参数
        self.norm_params = {}
    
    def fit_normalization(self, x: torch.Tensor) -> None:
        """拟合标准化参数"""
        if self.normalization_method == 'standard':
            self.norm_params['mean'] = x.mean(dim=(0, 2), keepdim=True)
            self.norm_params['std'] = x.std(dim=(0, 2), keepdim=True) + 1e-8
        
        elif self.normalization_method == 'robust':
            # 使用中位数和四分位距
            flattened = x.flatten(0, 2)  # (batch*time, features)
            self.norm_params['median'] = torch.median(flattened, dim=0)[0].unsqueeze(0).unsqueeze(2)
            q75 = torch.quantile(flattened, 0.75, dim=0).unsqueeze(0).unsqueeze(2)
            q25 = torch.quantile(flattened, 0.25, dim=0).unsqueeze(0).unsqueeze(2)
            self.norm_params['iqr'] = (q75 - q25) + 1e-8
        
        elif self.normalization_method == 'minmax':
            flattened = x.flatten(0, 2)
            self.norm_params['min'] = torch.min(flattened, dim=0)[0].unsqueeze(0).unsqueeze(2)
            self.norm_params['max'] = torch.max(flattened, dim=0)[0].unsqueeze(0).unsqueeze(2)
            self.norm_params['range'] = self.norm_params['max'] - self.norm_params['min'] + 1e-8
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """应用标准化"""
        if self.normalization_method == 'standard':
            return (x - self.norm_params['mean']) / self.norm_params['std']
        
        elif self.normalization_method == 'robust':
            return (x - self.norm_params['median']) / self.norm_params['iqr']
        
        elif self.normalization_method == 'minmax':
            return (x - self.norm_params['min']) / self.norm_params['range']
        
        else:
            return x
    
    def handle_missing_values(self, x: torch.Tensor) -> torch.Tensor:
        """处理缺失值"""
        if self.handle_missing == 'interpolate':
            # 线性插值
            mask = torch.isnan(x)
            if mask.any():
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        for k in range(x.size(3)):
                            series = x[i, j, :, k]
                            if torch.isnan(series).any():
                                # 简单的前向填充和后向填充
                                valid_indices = ~torch.isnan(series)
                                if valid_indices.any():
                                    # 前向填充
                                    last_valid = None
                                    for t in range(len(series)):
                                        if valid_indices[t]:
                                            last_valid = series[t]
                                        elif last_valid is not None:
                                            series[t] = last_valid
                                    
                                    # 后向填充开头的缺失值
                                    first_valid = None
                                    for t in range(len(series)-1, -1, -1):
                                        if valid_indices[t]:
                                            first_valid = series[t]
                                        elif first_valid is not None and torch.isnan(series[t]):
                                            series[t] = first_valid
        
        elif self.handle_missing == 'zero':
            x = torch.nan_to_num(x, nan=0.0)
        
        elif self.handle_missing == 'mean':
            # 用每个特征的均值填充
            for i in range(x.size(3)):
                feature_data = x[:, :, :, i]
                mean_val = feature_data[~torch.isnan(feature_data)].mean()
                x[:, :, :, i] = torch.nan_to_num(x[:, :, :, i], nan=mean_val.item())
        
        return x
    
    def remove_outliers_iqr(self, x: torch.Tensor) -> torch.Tensor:
        """使用IQR方法移除异常值"""
        if not self.remove_outliers:
            return x
        
        for i in range(x.size(3)):  # 对每个特征
            feature_data = x[:, :, :, i].flatten()
            q75 = torch.quantile(feature_data, 0.75)
            q25 = torch.quantile(feature_data, 0.25)
            iqr = q75 - q25
            
            lower_bound = q25 - self.outlier_threshold * iqr
            upper_bound = q75 + self.outlier_threshold * iqr
            
            # 将异常值替换为边界值
            x[:, :, :, i] = torch.clamp(x[:, :, :, i], lower_bound, upper_bound)
        
        return x
    
    def add_technical_indicators(self, x: torch.Tensor) -> torch.Tensor:
        """添加技术指标特征"""
        if not self.feature_engineering:
            return x
        
        # 假设输入的最后几个维度是 [open, high, low, close, volume]
        # 这里添加一些简单的技术指标
        
        batch_size, channels, time_steps, features = x.shape
        additional_features = []
        
        # 移动平均
        close_prices = x[:, :, :, 3]  # 假设收盘价在第4个位置
        ma_5 = self._moving_average(close_prices, 5)
        ma_10 = self._moving_average(close_prices, 10)
        additional_features.extend([ma_5, ma_10])
        
        # RSI (简化版)
        rsi = self._calculate_rsi(close_prices)
        additional_features.append(rsi)
        
        # 价格变化率
        price_change = (close_prices[:, :, 1:] - close_prices[:, :, :-1]) / (close_prices[:, :, :-1] + 1e-8)
        price_change = torch.cat([torch.zeros_like(close_prices[:, :, :1]), price_change], dim=2)
        additional_features.append(price_change)
        
        # 将新特征添加到原始数据
        if additional_features:
            additional_features = torch.stack(additional_features, dim=-1)
            x = torch.cat([x, additional_features], dim=-1)
        
        return x
    
    def _moving_average(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """计算移动平均"""
        if window >= x.size(2):
            return x.mean(dim=2, keepdim=True).repeat(1, 1, x.size(2))
        
        # 使用卷积计算移动平均
        kernel = torch.ones(1, 1, window, device=x.device) / window
        padded = torch.nn.functional.pad(x, (0, 0, window-1, 0), mode='replicate')
        ma = torch.nn.functional.conv1d(padded.view(-1, 1, x.size(2) + window - 1), kernel)
        return ma.view(x.shape)
    
    def _calculate_rsi(self, prices: torch.Tensor, period: int = 14) -> torch.Tensor:
        """计算RSI（简化版）"""
        # 价格变化
        deltas = prices[:, :, 1:] - prices[:, :, :-1]
        deltas = torch.cat([torch.zeros_like(prices[:, :, :1]), deltas], dim=2)
        
        # 分离涨跌
        gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
        
        # 计算平均涨跌
        avg_gains = self._moving_average(gains, period)
        avg_losses = self._moving_average(losses, period)
        
        # RSI
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def process(self, x: torch.Tensor, fit: bool = False) -> torch.Tensor:
        """完整的预处理流程"""
        # 处理缺失值
        x = self.handle_missing_values(x)
        
        # 移除异常值
        x = self.remove_outliers_iqr(x)
        
        # 添加技术指标
        x = self.add_technical_indicators(x)
        
        # 标准化
        if fit:
            self.fit_normalization(x)
        x = self.normalize(x)
        
        return x


class AdaptiveLossWeighting:
    """
    自适应损失权重调整
    
    根据训练进度和类别分布动态调整损失权重
    """
    
    def __init__(self, 
                 initial_weights: Optional[torch.Tensor] = None,
                 update_frequency: int = 100,
                 momentum: float = 0.9):
        
        self.initial_weights = initial_weights
        self.update_frequency = update_frequency
        self.momentum = momentum
        
        self.current_weights = initial_weights
        self.step_count = 0
        self.class_counts = None
    
    def update_weights(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """更新类别权重"""
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            # 计算当前批次的类别分布
            num_classes = y_pred.size(1)
            batch_counts = torch.bincount(y_true, minlength=num_classes).float()
            
            if self.class_counts is None:
                self.class_counts = batch_counts
            else:
                # 使用动量更新
                self.class_counts = self.momentum * self.class_counts + (1 - self.momentum) * batch_counts
            
            # 计算新的权重（逆频率）
            total_samples = self.class_counts.sum()
            new_weights = total_samples / (num_classes * self.class_counts + 1e-8)
            
            if self.current_weights is None:
                self.current_weights = new_weights
            else:
                self.current_weights = self.momentum * self.current_weights + (1 - self.momentum) * new_weights
        
        return self.current_weights if self.current_weights is not None else torch.ones(y_pred.size(1))
    
    def elastic_deformation(self, x: torch.Tensor, alpha: float = 1.0, sigma: float = 50.0) -> torch.Tensor:
        """弹性变形（适合时序数据）"""
        batch_size, channels, time_steps, features = x.shape
        
        # 生成随机位移场
        dx = torch.randn(batch_size, 1, time_steps, 1, device=x.device) * alpha
        dy = torch.randn(batch_size, 1, 1, features, device=x.device) * alpha
        
        # 应用高斯平滑
        dx = torch.nn.functional.avg_pool2d(dx, kernel_size=sigma, stride=1, padding=sigma//2)
        dy = torch.nn.functional.avg_pool2d(dy, kernel_size=sigma, stride=1, padding=sigma//2)
        
        # 创建网格
        time_grid, feature_grid = torch.meshgrid(
            torch.arange(time_steps, device=x.device),
            torch.arange(features, device=x.device),
            indexing='ij'
        )
        
        # 应用位移
        time_coords = time_grid + dx[0, 0, :, :]
        feature_coords = feature_grid + dy[0, 0, :, :]
        
        # 确保坐标在有效范围内
        time_coords = torch.clamp(time_coords, 0, time_steps - 1)
        feature_coords = torch.clamp(feature_coords, 0, features - 1)
        
        # 双线性插值
        x_deformed = torch.nn.functional.grid_sample(
            x.view(batch_size * channels, 1, time_steps, features),
            torch.stack([feature_coords, time_coords], dim=-1).unsqueeze(0).repeat(batch_size * channels, 1, 1, 1),
            mode='bilinear',
            align_corners=False
        )
        
        return x_deformed.view(batch_size, channels, time_steps, features)
    
    def apply_augmentation_pipeline(self, x: torch.Tensor, augmentation_prob: float = 0.5) -> torch.Tensor:
        """
        应用完整的增强流水线
        
        Args:
            x: 输入张量
            augmentation_prob: 每个增强方法的应用概率
            
        Returns:
            增强后的张量
        """
        augmented = x.clone()
        
        # 随机选择增强方法
        augmentations = [
            (self.add_noise, {}),
            (self.time_shift, {}),
            (self.magnitude_warp, {}),
            (self.add_jitter, {}),
            (self.scale_magnitude, {}),
            (self.random_masking, {}),
            (self.frequency_dropout, {})
        ]
        
        for aug_func, kwargs in augmentations:
            if torch.rand(1) < augmentation_prob:
                try:
                    augmented = aug_func(augmented, **kwargs)
                except Exception as e:
                    logger.warning(f"增强方法 {aug_func.__name__} 失败: {e}")
                    continue
        
        return augmented