"""
训练稳定性配置

这个文件包含防止训练过程中出现NaN值的各种配置选项
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class StabilityConfig:
    """训练稳定性配置"""
    
    # 梯度裁剪
    max_grad_norm: float = 0.5  # 降低梯度裁剪阈值
    
    # 学习率调整
    lr_reduction_factor: float = 0.5  # NaN检测时的学习率减少因子
    min_lr: float = 1e-6  # 最小学习率
    
    # 权重检查
    check_weights_every: int = 1  # 每N个batch检查一次权重
    reinitialize_on_nan: bool = True  # 检测到NaN时重新初始化权重
    
    # 数据验证
    validate_input_data: bool = True  # 验证输入数据
    skip_invalid_batches: bool = True  # 跳过无效的batch
    
    # 数据增强安全
    safe_augmentation: bool = True  # 安全的数据增强
    max_augmentation_attempts: int = 3  # 最大增强尝试次数
    
    # 数值稳定性
    use_amp: bool = True  # 使用混合精度训练
    use_gradient_scaling: bool = True  # 使用梯度缩放
    
    # 早停和恢复
    early_stop_on_nan: bool = True  # NaN时早停
    max_nan_batches: int = 10  # 最大连续NaN batch数
    restore_checkpoint_on_nan: bool = True  # NaN时恢复检查点

# 默认配置
DEFAULT_STABILITY_CONFIG = StabilityConfig()

# 高稳定性配置（更保守）
HIGH_STABILITY_CONFIG = StabilityConfig(
    max_grad_norm=0.1,
    lr_reduction_factor=0.3,
    min_lr=1e-7,
    check_weights_every=1,
    max_augmentation_attempts=1,
    max_nan_batches=5
)

# 平衡配置
BALANCED_STABILITY_CONFIG = StabilityConfig(
    max_grad_norm=0.5,
    lr_reduction_factor=0.5,
    min_lr=1e-6,
    check_weights_every=1,
    max_augmentation_attempts=2,
    max_nan_batches=10
)
