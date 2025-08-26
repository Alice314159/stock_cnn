#!/usr/bin/env python3
"""
多股票CNN-LSTM模型训练脚本

功能:
1. 读取指定文件夹下的多只股票数据
2. 为每只股票独立训练Scaler
3. 使用CNN-LSTM混合模型进行训练
4. 支持多GPU训练和混合精度训练
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict,List, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger
import pickle

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.config import ModelConfig, TrainingConfig, DataConfig, ExperimentConfig
from src.models.cnn_lstm import ImprovedCNNLSTMModel, EfficientMultiStockModel
from src.data.multi_stock_processor import MultiStockProcessor
from src.data.multi_stock_dataset import MultiStockSequenceDataset, MultiStockDataLoader
from src.data.augmentation import StockDataAugmentation


def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_config: str) -> torch.device:
    """获取设备"""
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("使用CPU")
    else:
        device = torch.device(device_config)
    
    return device


def create_model(config: ModelConfig, num_features: int, num_stocks: int) -> nn.Module:
    """创建模型"""
    if num_stocks > 1:
        # 多股票模型
        model = EfficientMultiStockModel(
            num_stocks=num_stocks,
            num_classes=2,
            base_channels=config.base_channels,
            channel_multiplier=config.channel_multiplier,
            num_blocks=config.num_blocks,
            dropout=config.dropout_rate,
            norm=config.norm_type,
            activation=config.activation,
            attention=config.attention_type,
            use_lstm=config.use_lstm,
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            lstm_dropout=config.lstm_dropout,
            lstm_bidirectional=config.lstm_bidirectional,
            window_size=config.window_size if hasattr(config, 'window_size') else 60,
            num_features=num_features
        )
        logger.info("创建多股票CNN-LSTM模型")
    else:
        # 单股票模型
        model = ImprovedCNNLSTMModel(
            num_classes=2,
            base_channels=config.base_channels,
            channel_multiplier=config.channel_multiplier,
            num_blocks=config.num_blocks,
            dropout=config.dropout_rate,
            norm=config.norm_type,
            activation=config.activation,
            attention=config.attention_type,
            use_depthwise_separable=config.use_depthwise_separable,
            use_lstm=config.use_lstm,
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            lstm_dropout=config.lstm_dropout,
            lstm_bidirectional=config.lstm_bidirectional,
            window_size=config.window_size if hasattr(config, 'window_size') else 60,
            num_features=num_features
        )
        logger.info("创建单股票CNN-LSTM模型")
    
    return model


def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """创建优化器（支持参数分组，norm和bias层不使用权重衰减）"""
    if config.optimizer == "adamw":
        # 参数分组：norm和bias层不使用权重衰减
        param_groups = []
        
        # 需要权重衰减的参数
        decay_params = []
        # 不需要权重衰减的参数
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 检查是否为norm层或bias参数
            if len(param.shape) == 1 or any(norm in name.lower() for norm in ['norm', 'bn', 'ln', 'gn']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # 添加参数组
        if decay_params:
            param_groups.append({
                'params': decay_params,
                'weight_decay': config.weight_decay,
                'lr': config.learning_rate
            })
        
        if no_decay_params:
            param_groups.append({
                'params': no_decay_params,
                'weight_decay': 0.0,
                'lr': config.learning_rate
            })
        
        optimizer = optim.AdamW(param_groups)
        logger.info(f"AdamW优化器: {len(decay_params)}个参数使用权重衰减, {len(no_decay_params)}个参数不使用权重衰减")
        
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"不支持的优化器: {config.optimizer}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: TrainingConfig, num_steps: int) -> Optional[optim.lr_scheduler._LRScheduler]:
    """创建学习率调度器"""
    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_steps,
            eta_min=config.learning_rate * config.min_lr_factor
        )
    elif config.scheduler == "cosine_warmup":
        # 带warmup的余弦退火
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        # 计算warmup步数
        warmup_steps = int(num_steps * config.warmup_epochs / config.epochs)
        
        # 创建warmup调度器
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        
        # 创建余弦退火调度器
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_steps - warmup_steps,
            eta_min=config.learning_rate * config.min_lr_factor
        )
        
        # 组合调度器
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        logger.info(f"创建带warmup的余弦退火调度器: warmup={warmup_steps}步, 总步数={num_steps}")
        
    elif config.scheduler == "onecycle":
        # OneCycleLR调度器
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=num_steps,
            pct_start=config.warmup_epochs / config.epochs,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        logger.info(f"创建OneCycleLR调度器: 最大学习率={config.learning_rate}, 总步数={num_steps}")
        
    elif config.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"不支持的学习率调度器: {config.scheduler}")
    
    return scheduler

def check_and_fix_model_weights(model: nn.Module) -> bool:
    """检查并修复模型权重中的NaN值"""
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.warning(f"参数 {name} 包含NaN或Inf值，重新初始化")
            # 重新初始化参数
            if 'weight' in name:
                if 'conv' in name or 'linear' in name:
                    nn.init.xavier_uniform_(param)
                elif 'norm' in name:
                    nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            has_nan = True
    return has_nan

def train_epoch(model, dataloader, criterion, optimizer, scaler, data_augmenter, device, epoch, 
                gradient_monitor, adaptive_clipper, accumulation_steps):
    """训练一个epoch，支持梯度累积"""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    # 梯度累积相关
    optimizer.zero_grad()
    accumulation_counter = 0
    
    for batch_idx, (data, stock_indices, targets) in enumerate(dataloader):
        # 数据验证
        if torch.isnan(data).any() or torch.isinf(data).any():
            logger.warning(f"Batch {batch_idx}: 输入数据包含NaN或Inf值，跳过此batch")
            continue
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            logger.warning(f"Batch {batch_idx}: 目标数据包含NaN或Inf值，跳过此batch")
            continue
        
        # 数据增强
        if data_augmenter is not None and torch.rand(1).item() < 0.5:
            try:
                original_data = data.clone()
                data = data_augmenter.apply_augmentation_pipeline(data, augmentation_prob=0.3)
                if torch.isnan(data).any() or torch.isinf(data).any():
                    logger.warning(f"Batch {batch_idx}: 数据增强后产生无效值，使用原始数据")
                    data = original_data
            except Exception as e:
                logger.warning(f"数据增强失败: {e}")
                data = original_data
        
        # 移动到设备
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        stock_indices = stock_indices.to(device, non_blocking=True)
        
        # 前向传播
        try:
            outputs = model(data, stock_indices)
        except ValueError as e:
            logger.error(f"模型前向传播失败: {e}")
            continue
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 检查损失值
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Loss异常: {loss.item()}")
            continue
        
        # 反向传播 - 支持梯度累积
        if scaler is not None:
            # 混合精度训练
            scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
        else:
            # 标准训练
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
        
        # 累积梯度
        accumulation_counter += 1
        
        if accumulation_counter >= accumulation_steps:
            # 执行参数更新
            if scaler is not None:
                # 自适应梯度裁剪
                adaptive_clipper.clip_gradients(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 自适应梯度裁剪
                adaptive_clipper.clip_gradients(model)
                optimizer.step()
            
            # 更新梯度监控
            gradient_stats = gradient_monitor.update(model, batch_idx)
            
            # 重置梯度
            optimizer.zero_grad()
            accumulation_counter = 0
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()
        
        # 打印进度
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, "
                       f"Loss: {loss.item():.4f}, Acc: {100*correct_predictions/total_predictions:.2f}%")
    
    # 处理剩余的梯度
    if accumulation_counter > 0:
        if scaler is not None:
            adaptive_clipper.clip_gradients(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            adaptive_clipper.clip_gradients(model)
            optimizer.step()
        optimizer.zero_grad()
    
    # 计算平均指标
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return avg_loss, {'loss': avg_loss, 'accuracy': accuracy}


 
def validate_epoch(model, dataloader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (data, stock_indices, targets) in enumerate(dataloader):
            # 数据验证
            if torch.isnan(data).any() or torch.isinf(data).any():
                logger.warning(f"验证Batch {batch_idx}: 输入数据包含NaN或Inf值，跳过此batch")
                continue
            
            # 移动到设备
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            stock_indices = stock_indices.to(device, non_blocking=True)
            
            # 前向传播
            try:
                outputs = model(data, stock_indices)
            except ValueError as e:
                logger.error(f"验证时模型前向传播失败: {e}")
                continue
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
    
    # 计算平均指标
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return avg_loss, {'loss': avg_loss, 'accuracy': accuracy}


def save_checkpoint(model, optimizer, epoch, loss, score, checkpoint_dir, name):
    """保存检查点"""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'score': score,
        'timestamp': datetime.now().isoformat()
    }
    
    file_path = checkpoint_path / f"{name}_model.pth"
    torch.save(checkpoint, file_path)
    logger.info(f"检查点已保存: {file_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    score = checkpoint['score']
    logger.info(f"检查点已加载: epoch {epoch}, score {score}")
    return epoch, score


def main():
    
    """主函数"""
    parser = argparse.ArgumentParser(description="多股票CNN-LSTM模型训练")
    parser.add_argument("--config", type=str, default="", help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--num-workers", type=int, default=None, help="数据加载器工作进程数 (Windows用户建议设为0)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建配置
    model_config = ModelConfig()
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    data_config = DataConfig()
    
    # 如果指定了num_workers，使用命令行参数；否则使用默认配置
    num_workers = args.num_workers if args.num_workers is not None else ExperimentConfig.num_workers
    experiment_config = ExperimentConfig(device=args.device, seed=args.seed, num_workers=num_workers)
    
    # 获取设备
    device = get_device(experiment_config.device)
    
    # 创建日志目录
    log_dir = Path(experiment_config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger.add(
        log_dir / "multi_stock_cnn_lstm_{time}.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info("开始多股票CNN-LSTM模型训练")
    logger.info(f"配置: {model_config}")
    logger.info(f"训练配置: {training_config}")
    logger.info(f"数据配置: {data_config}")
    logger.info(f"实验配置: {experiment_config}")
    
    # 关于num_workers的提示
    if experiment_config.num_workers == 0:
        logger.info("注意: 使用0个工作进程 (num_workers=0)，适合Windows用户或调试")
    else:
        logger.info(f"使用{experiment_config.num_workers}个工作进程进行数据加载")
    
    try:
        # 数据预处理
        logger.info("开始数据预处理...")
        
        # 预处理器配置
        preprocessor_config = data_config.preprocessor_config
        if preprocessor_config is None:
            preprocessor_config = {
                "scaler_type": "robust",  # 对异常值更鲁棒
                "feature_selection_method": data_config.feature_selection,
                "max_features": data_config.max_features,
                "outlier_detection": True,
                "outlier_threshold": 3.0,
                "fill_missing": "interpolate",
                "min_data_quality": 0.8
            }
        
        processor = MultiStockProcessor(
            data_folder=data_config.data_folder,
            stock_file_pattern=data_config.stock_file_pattern,
            min_stock_count=data_config.min_stock_count,
            window_size=data_config.window_size,
            stride=data_config.stride,
            prediction_horizon=data_config.prediction_horizon,
            binary_threshold=data_config.binary_threshold,
            max_features=data_config.max_features,
            feature_selection=data_config.feature_selection,
            preprocessor_config=preprocessor_config,
            use_advanced_preprocessing=data_config.use_advanced_preprocessing
        )
        
        processor.process_all_stocks()
        
        # 数据分割
        train_data, val_data, test_data = processor.split_data(
            train_ratio=data_config.train_ratio,
            val_ratio=data_config.val_ratio
        )
        
        # 获取特征列
        feature_columns = processor.get_feature_columns()
        num_features = len(feature_columns)
        num_stocks = len(processor.stock_data)
        
        logger.info(f"特征数量: {num_features}")
        logger.info(f"股票数量: {num_stocks}")
        logger.info(f"训练集大小: {len(train_data)}")
        logger.info(f"验证集大小: {len(val_data)}")
        logger.info(f"测试集大小: {len(test_data)}")
        
        # 检查窗口大小是否合适
        min_dataset_size = min(len(train_data), len(val_data), len(test_data))
        if data_config.window_size > min_dataset_size:
            logger.warning(f"窗口大小({data_config.window_size})大于最小数据集大小({min_dataset_size})")
            logger.warning("建议减小窗口大小或增加数据量")
            # 自动调整窗口大小
            adjusted_window_size = max(10, min_dataset_size // 4)  # 至少保留10个样本，最多使用1/4的数据
            logger.info(f"自动调整窗口大小为: {adjusted_window_size}")
            data_config.window_size = adjusted_window_size
        
        # 创建数据集
        train_dataset = MultiStockSequenceDataset(
            train_data, feature_columns, data_config.window_size, data_config.stride,
            stock_idx_column="stock_index", label_column="label",
            use_lazy_loading=True  # 启用延迟加载以节省内存
        )
        val_dataset = MultiStockSequenceDataset(
            val_data, feature_columns, data_config.window_size, data_config.stride,
            stock_idx_column="stock_index", label_column="label",
            use_lazy_loading=True
        )
        test_dataset = MultiStockSequenceDataset(
            test_data, feature_columns, data_config.window_size, data_config.stride,
            stock_idx_column="stock_index", label_column="label",
            use_lazy_loading=True
        )
        
        # 数据增强配置
        augmentation_config = data_config.augmentation_config
        if augmentation_config is None:
            augmentation_config = {
                "noise_std": 0.01,
                "time_shift_max": 3,
                "magnitude_warp_sigma": 0.1,
                "jitter_std": 0.005,
                "scaling_factor": 0.05,
                "masking_ratio": 0.05
            }
        
        # 创建数据增强器
        if data_config.use_data_augmentation:
            data_augmenter = StockDataAugmentation(**augmentation_config)
            logger.info("启用数据增强功能")
        else:
            data_augmenter = None
            logger.info("禁用数据增强功能")
        
        # 创建数据加载器 - 使用新的优化选项
        train_loader = MultiStockDataLoader(
            train_dataset, training_config.batch_size, shuffle=True,
            num_workers=experiment_config.num_workers, 
            pin_memory=experiment_config.pin_memory,
            persistent_workers=True,  # 启用持久化workers
            prefetch_factor=2,        # 预取因子
            non_blocking=True         # 非阻塞传输
        )
        val_loader = MultiStockDataLoader(
            val_dataset, training_config.batch_size, shuffle=False,
            num_workers=experiment_config.num_workers, 
            pin_memory=experiment_config.pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            non_blocking=True
        )
        test_loader = MultiStockDataLoader(
            test_dataset, training_config.batch_size, shuffle=False,
            num_workers=experiment_config.num_workers, 
            pin_memory=experiment_config.pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            non_blocking=True
        )
        
        # 创建模型
        model = create_model(model_config, num_features, num_stocks)
        model = model.to(device)
        
        # 使用torch.compile优化模型（PyTorch 2.x特性）
        try:
            if hasattr(torch, 'compile') and False:  # 临时禁用torch.compile
                logger.info("使用torch.compile优化模型...")
                # 在Windows上，torch.compile可能需要C++编译器，使用更安全的配置
                if device.type == "cpu":
                    # CPU模式使用更简单的编译选项
                    model = torch.compile(model, mode="reduce-overhead", backend="inductor")
                else:
                    # GPU模式
                    model = torch.compile(model, mode="reduce-overhead")
                logger.info("模型编译成功，启用优化")
            else:
                logger.info("跳过torch.compile优化（已临时禁用）")
        except Exception as e:
            logger.warning(f"torch.compile失败，使用原始模型: {e}")
            logger.info("这通常是因为缺少C++编译器（Windows上需要安装Visual Studio Build Tools）")
            logger.info("模型将继续使用标准模式运行")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
        # 创建损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 创建优化器
        optimizer = create_optimizer(model, training_config)
        
        # 创建学习率调度器
        scheduler = create_scheduler(optimizer, training_config, len(train_loader) * training_config.epochs)
        
        # 创建混合精度训练器
        scaler = GradScaler() if training_config.use_amp else None
        
        # 创建梯度监控器和自适应裁剪器
        from src.training.gradient_monitor import GradientMonitor, AdaptiveGradientClipper
        
        gradient_monitor = GradientMonitor(
            history_size=100,
            spike_threshold=10.0,
            vanish_threshold=1e-8
        )
        
        adaptive_clipper = AdaptiveGradientClipper(
            initial_norm=0.5,
            percentile=95.0,
            min_norm=0.1,
            max_norm=5.0
        )
        
        # 梯度累积配置
        accumulation_steps = 4  # 每4个batch更新一次参数
        effective_batch_size = training_config.batch_size * accumulation_steps
        logger.info(f"梯度累积: {accumulation_steps}步, 有效批次大小: {effective_batch_size}")
        
        # 训练循环
        best_val_score = 0.0
        patience_counter = 0
        
        logger.info("开始训练...")
        
        for epoch in range(training_config.epochs):
            # 训练阶段
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, scaler, 
                data_augmenter, device, epoch, 
                gradient_monitor, adaptive_clipper, accumulation_steps
            )
            
            # 验证阶段
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch
            )
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step()
            
            # 记录指标
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{training_config.epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"LR: {current_lr:.6f}")
            
            # 早停检查
            if val_metrics['accuracy'] > best_val_score:
                best_val_score = val_metrics['accuracy']
                patience_counter = 0
                # 保存最佳模型
                save_checkpoint(model, optimizer, epoch, val_loss, best_val_score, 
                              experiment_config.checkpoint_dir, "best")
            else:
                patience_counter += 1
                if patience_counter >= training_config.patience:
                    logger.info(f"早停触发，{training_config.patience}个epoch无改善")
                    break
            
            # 定期保存检查点
            if (epoch + 1) % experiment_config.save_interval == 0:
                save_checkpoint(model, optimizer, epoch, val_loss, val_metrics['accuracy'], 
                              experiment_config.checkpoint_dir, f"epoch_{epoch+1}")
            
            # 打印梯度监控建议
            if epoch % 5 == 0:  # 每5个epoch打印一次
                recommendations = gradient_monitor.get_recommendations()
                if recommendations:
                    logger.info("梯度监控建议:")
                    for rec in recommendations:
                        logger.info(f"  - {rec}")
        
        # 测试最佳模型
        logger.info("加载最佳模型进行测试...")
        checkpoint_dir = Path(experiment_config.checkpoint_dir)
        best_checkpoint_path = checkpoint_dir / "best_model.pth"
        
        if best_checkpoint_path.exists():
            epoch, score = load_checkpoint(model, optimizer, best_checkpoint_path)
            logger.info(f"加载最佳模型检查点: epoch {epoch}, score {score}")
        else:
            logger.warning(f"最佳模型检查点文件不存在: {best_checkpoint_path}")
            # 尝试加载最新的检查点
            epoch_checkpoints = list(checkpoint_dir.glob("epoch_*_model.pth"))
            if epoch_checkpoints:
                latest_checkpoint_path = sorted(epoch_checkpoints)[-1]
                epoch, score = load_checkpoint(model, optimizer, latest_checkpoint_path)
                logger.info(f"加载最新的检查点: {latest_checkpoint_path}")
            else:
                logger.warning("没有找到任何检查点文件，使用当前模型进行测试")
        
        test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device, epoch)
        logger.info(f"测试结果 - Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']:.2f}%")
        
        # 保存测试结果
        results = {
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'best_val_score': best_val_score,
            'final_epoch': epoch,
            'training_config': training_config,
            'model_config': model_config,
            'data_config': data_config
        }
        
        # 保存结果
        results_path = Path(experiment_config.results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        with open(results_path / "training_results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        logger.info("训练完成！结果已保存")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
