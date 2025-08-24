from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from .losses import FocalLoss, LabelSmoothingCE
from .validation import compute_metrics
import logging
import time
from collections import defaultdict

class OptimizedTrainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.cfg = config
        self.scaler = GradScaler(enabled=self.cfg.use_amp)
        
        # 优化器选择
        self.opt = self._build_optimizer()
        
        # 学习率调度器优化
        self.sched = self._build_scheduler()
        
        # 损失函数选择
        self.criterion = self._build_criterion()
        
        # 训练历史记录
        self.history = defaultdict(list)
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _build_optimizer(self):
        """构建优化器，支持更多配置"""
        params = self.model.parameters()
        
        if self.cfg.optimizer == "adamw":
            return AdamW(
                params, 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay,
                betas=getattr(self.cfg, 'betas', (0.9, 0.999)),
                eps=getattr(self.cfg, 'eps', 1e-8)
            )
        elif self.cfg.optimizer == "adam":
            return Adam(
                params, 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay,
                betas=getattr(self.cfg, 'betas', (0.9, 0.999))
            )
        else:
            return SGD(
                params, 
                lr=self.cfg.learning_rate, 
                momentum=getattr(self.cfg, 'momentum', 0.9), 
                weight_decay=self.cfg.weight_decay,
                nesterov=getattr(self.cfg, 'nesterov', True)
            )

    def _build_scheduler(self):
        """构建学习率调度器"""
        if not hasattr(self.cfg, 'scheduler') or self.cfg.scheduler is None:
            return None
            
        if self.cfg.scheduler == "cosine":
            return CosineAnnealingLR(
                self.opt, 
                T_max=self.cfg.epochs,
                eta_min=getattr(self.cfg, 'min_lr', 1e-6)
            )
        elif self.cfg.scheduler == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.opt,
                mode='max',
                factor=getattr(self.cfg, 'lr_factor', 0.5),
                patience=getattr(self.cfg, 'lr_patience', 5),
                min_lr=getattr(self.cfg, 'min_lr', 1e-6)
            )
        elif self.cfg.scheduler == "one_cycle":
            # 需要知道总的step数
            steps_per_epoch = getattr(self.cfg, 'steps_per_epoch', 100)
            return OneCycleLR(
                self.opt,
                max_lr=self.cfg.learning_rate,
                epochs=self.cfg.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=getattr(self.cfg, 'pct_start', 0.3)
            )
        return None

    def _build_criterion(self):
        """构建损失函数"""
        loss_type = getattr(self.cfg, 'loss_type', 'label_smoothing')
        
        if loss_type == 'focal':
            return FocalLoss(
                alpha=getattr(self.cfg, 'focal_alpha', 1.0),
                gamma=getattr(self.cfg, 'focal_gamma', 2.0),
                reduction='mean'
            )
        elif loss_type == 'label_smoothing':
            return LabelSmoothingCE(
                smoothing=getattr(self.cfg, 'label_smoothing', 0.05)
            )
        else:
            return torch.nn.CrossEntropyLoss()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """优化的训练循环"""
        best = {self.cfg.monitor: -1e9 if 'acc' in self.cfg.monitor or 'f1' in self.cfg.monitor else 1e9}
        patience = 0
        
        # 为OneCycleLR设置步数
        if hasattr(self.cfg, 'scheduler') and self.cfg.scheduler == 'one_cycle':
            self.cfg.steps_per_epoch = len(train_loader)
            self.sched = self._build_scheduler()
        
        for epoch in range(self.cfg.epochs):
            epoch_start = time.time()
            
            # 训练阶段
            train_loss = self._train_epoch(train_loader, epoch)
            
            # 验证阶段
            val_metrics = self._validate_epoch(val_loader)
            
            # 学习率调度
            if self.sched is not None:
                if isinstance(self.sched, ReduceLROnPlateau):
                    self.sched.step(val_metrics[self.cfg.monitor])
                elif not isinstance(self.sched, OneCycleLR):
                    self.sched.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            for k, v in val_metrics.items():
                self.history[k].append(v)
            
            epoch_time = time.time() - epoch_start
            
            # 日志输出
            self.logger.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Acc: {val_metrics.get('val_acc', 0):.4f} - "
                f"Val F1: {val_metrics.get('val_f1_macro', 0):.4f} - "
                f"Time: {epoch_time:.2f}s - "
                f"LR: {self.opt.param_groups[0]['lr']:.2e}"
            )
            
            # 早停和模型保存
            monitor_value = val_metrics.get(self.cfg.monitor, None)
            if monitor_value is not None:
                is_better = (monitor_value > best[self.cfg.monitor] 
                           if 'acc' in self.cfg.monitor or 'f1' in self.cfg.monitor 
                           else monitor_value < best[self.cfg.monitor])
                
                if is_better:
                    best.update(val_metrics)
                    patience = 0
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'scheduler_state_dict': self.sched.state_dict() if self.sched else None,
                        'epoch': epoch,
                        'best_metrics': best
                    }, "checkpoints/best.pt")
                    self.logger.info(f"New best model saved with {self.cfg.monitor}: {monitor_value:.4f}")
                else:
                    patience += 1
                    
                if patience >= self.cfg.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return best, self.history

    def _train_epoch(self, train_loader: DataLoader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(self.device), yb.to(self.device)
            
            with autocast(enabled=self.cfg.use_amp):
                out = self.model(xb)
                loss = self.criterion(out, yb)
                if hasattr(loss, 'mean'):
                    loss = loss.mean()

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if hasattr(self.cfg, 'grad_clip') and self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            
            self.scaler.step(self.opt)
            self.scaler.update()
            
            # OneCycleLR需要每个batch更新
            if self.sched and isinstance(self.sched, OneCycleLR):
                self.sched.step()
            
            total_loss += loss.item()
            
            # 可选：打印batch级别的进度
            if hasattr(self.cfg, 'log_batch_freq') and batch_idx % self.cfg.log_batch_freq == 0:
                self.logger.info(
                    f"Epoch {epoch+1} Batch {batch_idx}/{num_batches} - "
                    f"Loss: {loss.item():.4f}"
                )
        
        return total_loss / num_batches

    def _validate_epoch(self, val_loader: DataLoader):
        """验证一个epoch"""
        self.model.eval()
        preds, tgts = [], []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                out = self.model(xb)
                preds.extend(out.argmax(1).cpu().numpy().tolist())
                tgts.extend(yb.numpy().tolist())
        
        return compute_metrics(tgts, preds)

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.sched and checkpoint.get('scheduler_state_dict'):
            self.sched.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('best_metrics', {})