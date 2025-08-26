"""
训练稳定性配置

这个文件包含防止训练过程中出现NaN值的各种配置选项
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StabilityConfig:
    """训练稳定性配置"""
    
    # 梯度裁剪配置
    gradient_clip_norm: float = 0.5
    adaptive_gradient_clipping: bool = True
    gradient_clip_percentile: float = 95.0
    
    # 学习率调整配置
    lr_reduction_factor: float = 0.5
    lr_reduction_patience: int = 3
    min_lr_factor: float = 0.01
    
    # 权重检查和修复配置
    check_weights_frequency: int = 10  # 每N个batch检查一次
    weight_reinit_threshold: float = 1e6  # 权重绝对值超过此值时重新初始化
    gradual_weight_fixing: bool = True  # 渐进式权重修复
    
    # 数据验证配置
    validate_input_data: bool = True
    validate_output_data: bool = True
    skip_invalid_batches: bool = True
    
    # 数据增强安全配置
    safe_augmentation: bool = True
    augmentation_fallback: bool = True
    
    # 混合精度训练配置
    use_amp: bool = True
    amp_scaler_growth_factor: float = 2.0
    amp_scaler_backoff_factor: float = 0.5
    
    # 梯度累积配置
    gradient_accumulation_steps: int = 1
    effective_batch_size: Optional[int] = None  # 如果设置，自动计算accumulation_steps
    
    # 智能梯度监控配置
    gradient_monitoring: bool = True
    gradient_norm_history_size: int = 100
    gradient_spike_threshold: float = 10.0  # 梯度范数超过此值时记录警告
    gradient_vanish_threshold: float = 1e-8  # 梯度范数低于此值时记录警告
    
    # 内存优化配置
    use_lazy_loading: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking_transfer: bool = True


# 预定义配置
DEFAULT = StabilityConfig()

HIGH = StabilityConfig(
    gradient_clip_norm=0.3,
    adaptive_gradient_clipping=True,
    gradient_clip_percentile=90.0,
    lr_reduction_factor=0.7,
    lr_reduction_patience=2,
    weight_reinit_threshold=1e5,
    gradient_spike_threshold=5.0,
    gradient_vanish_threshold=1e-6
)

BALANCED = StabilityConfig(
    gradient_clip_norm=0.5,
    adaptive_gradient_clipping=True,
    gradient_clip_percentile=95.0,
    lr_reduction_factor=0.5,
    lr_reduction_patience=3,
    weight_reinit_threshold=1e6,
    gradient_spike_threshold=10.0,
    gradient_vanish_threshold=1e-8
)

MEMORY_OPTIMIZED = StabilityConfig(
    gradient_clip_norm=0.5,
    adaptive_gradient_clipping=True,
    gradient_accumulation_steps=4,
    use_lazy_loading=True,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True,
    non_blocking_transfer=True
)
