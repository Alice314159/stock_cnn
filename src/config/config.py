
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

@dataclass
class ModelConfig:
    base_channels: int = 16
    channel_multiplier: float = 1.5
    num_blocks: int = 2
    dropout_rate: float = 0.3
    use_attention: bool = True
    attention_type: str = "se"  # "se", "eca", "cbam"
    norm_type: str = "group"     # "batch", "instance", "layer", "group"
    activation: str = "swish"    # "relu", "swish", "gelu", "mish"
    
    # 优化配置
    use_depthwise_separable: bool = True  # 使用深度可分离卷积减少计算量
    
    # LSTM配置
    use_lstm: bool = True
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = True

@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 基本训练参数
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 优化器配置
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    
    # 学习率调度器配置
    scheduler: str = "cosine"  # "cosine", "cosine_warmup", "onecycle", "none"
    warmup_epochs: int = 5
    min_lr_factor: float = 0.01
    
    # 早停配置
    patience: int = 8  # 早停耐心值
    early_stopping_patience: int = 8  # 保持向后兼容
    
    # 混合精度训练
    use_amp: bool = True
    
    # 梯度累积
    gradient_accumulation_steps: int = 1
    
    # 数据增强
    use_data_augmentation: bool = True
    
    # 监控指标
    monitor: str = "accuracy"  # 监控的指标
    monitor_mode: str = "max"  # "max" 或 "min"

@dataclass
class DataConfig:
    # 多股票数据配置
    data_folder: str = r"F:\VesperSet\stock_data_analysis\data\raw\HS300"  # 股票数据文件夹路径
    stock_file_pattern: str = "*.csv"  # 股票文件匹配模式
    min_stock_count: int = 2  # 最少需要的股票数量
    
    # 单股票配置
    csv_path: str = r"F:\VesperSet\stock_data_analysis\data\raw\indexData\HS300_daily_kline.csv"  # 保留向后兼容
    
    # 数据处理配置
    window_size: int = 60
    stride: int = 1
    prediction_horizon: int = 5
    binary_threshold: float = 0.005
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 特征工程配置
    max_features: int = 100  # 最大特征数量（增加以容纳更多技术指标）
    feature_selection: str = "correlation"  # "correlation", "mutual_info", "random_forest", "pca", "none"
    
    # 特征工程详细配置
    feature_engineering_config: Optional[Dict] = None  # 特征工程详细配置
    
    # 技术指标配置
    use_basic_indicators: bool = True      # 基础价格特征
    use_moving_averages: bool = True       # 移动平均线
    use_bollinger_bands: bool = True      # 布林带
    use_momentum_indicators: bool = True   # 动量指标
    use_oscillators: bool = True          # 振荡器指标
    use_volume_indicators: bool = True    # 成交量指标
    use_pattern_features: bool = True     # 价格模式特征
    use_statistical_features: bool = True # 统计特征
    use_trend_indicators: bool = True     # 趋势指标
    
    # 预处理器配置
    use_advanced_preprocessing: bool = True
    preprocessor_config: Optional[Dict] = None
    
    # 数据增强配置
    use_data_augmentation: bool = True
    augmentation_config: Optional[Dict] = None

@dataclass
class ExperimentConfig:
    """实验配置"""
    
    # 设备配置
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # 随机种子
    seed: int = 42
    
    # 数据加载器配置
    num_workers: int = 0  # Windows用户建议设为0
    pin_memory: bool = True
    
    # 日志配置
    log_dir: str = "logs"
    
    # 检查点配置
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5  # 每N个epoch保存一次检查点
    
    # 结果配置
    results_dir: str = "results"
    
    # 实验名称
    experiment_name: str = "multi_stock_cnn_lstm"
    
    # 是否启用TensorBoard
    use_tensorboard: bool = False
    
    # 是否启用WandB
    use_wandb: bool = False
    wandb_project: str = "stock_cnn_lstm"
    wandb_entity: str = ""
