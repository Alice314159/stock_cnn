import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

class EnsembleModel(nn.Module):
    """
    集成模型
    
    支持多种集成策略：
    1. 简单平均
    2. 加权平均
    3. 堆叠集成
    4. 动态集成
    """
    
    def __init__(self, 
                 models: List[nn.Module],
                 ensemble_method: str = 'weighted_average',
                 num_classes: int = 2,
                 meta_features_dim: int = 64):
        
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes
        self.num_models = len(models)
        
        if ensemble_method == 'weighted_average':
            # 可学习的权重
            self.model_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        
        elif ensemble_method == 'stacking':
            # 元学习器
            self.meta_learner = nn.Sequential(
                nn.Linear(self.num_models * num_classes, meta_features_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(meta_features_dim, meta_features_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(meta_features_dim // 2, num_classes)
            )
        
        elif ensemble_method == 'dynamic':
            # 动态权重网络
            self.attention_net = nn.Sequential(
                nn.Linear(self.num_models * num_classes, meta_features_dim),
                nn.ReLU(),
                nn.Linear(meta_features_dim, self.num_models),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 获取所有模型的预测
        model_outputs = []
        for model in self.models:
            with torch.no_grad() if self.ensemble_method != 'stacking' else torch.enable_grad():
                output = model(x)
                model_outputs.append(output)
        
        model_outputs = torch.stack(model_outputs, dim=1)  # (batch_size, num_models, num_classes)
        
        if self.ensemble_method == 'simple_average':
            return model_outputs.mean(dim=1)
        
        elif self.ensemble_method == 'weighted_average':
            weights = F.softmax(self.model_weights, dim=0)
            weighted_sum = torch.sum(model_outputs * weights.view(1, -1, 1), dim=1)
            return weighted_sum
        
        elif self.ensemble_method == 'stacking':
            # 将所有模型输出平铺作为元学习器输入
            meta_input = model_outputs.view(model_outputs.size(0), -1)
            return self.meta_learner(meta_input)
        
        elif self.ensemble_method == 'dynamic':
            # 动态权重
            meta_input = model_outputs.view(model_outputs.size(0), -1)
            attention_weights = self.attention_net(meta_input)  # (batch_size, num_models)
            
            # 应用注意力权重
            weighted_output = torch.sum(
                model_outputs * attention_weights.unsqueeze(-1), 
                dim=1
            )
            return weighted_output
        
        else:
            return model_outputs.mean(dim=1)
    
    def get_model_contributions(self, x: torch.Tensor) -> Dict[str, float]:
        """获取每个模型的贡献度"""
        model_outputs = []
        for i, model in enumerate(self.models):
            output = model(x)
            model_outputs.append(output)
        
        model_outputs = torch.stack(model_outputs, dim=1)
        
        if self.ensemble_method == 'weighted_average':
            weights = F.softmax(self.model_weights, dim=0)
            return {f'model_{i}': weight.item() for i, weight in enumerate(weights)}
        
        elif self.ensemble_method == 'dynamic':
            meta_input = model_outputs.view(model_outputs.size(0), -1)
            attention_weights = self.attention_net(meta_input)
            avg_weights = attention_weights.mean(dim=0)
            return {f'model_{i}': weight.item() for i, weight in enumerate(avg_weights)}
        
        else:
            return {f'model_{i}': 1.0 / self.num_models for i in range(self.num_models)}


class MetaLearningModel(nn.Module):
    """
    元学习模型（Model-Agnostic Meta-Learning简化版）
    
    用于快速适应新的股票或市场条件
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 meta_lr: float = 0.01,
                 inner_steps: int = 5,
                 adaptation_layers: Optional[List[str]] = None):
        
        super().__init__()
        
        self.base_model = base_model
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        
        # 确定哪些层参与适应
        if adaptation_layers is None:
            # 默认只适应分类头
            self.adaptation_params = {
                name: param for name, param in base_model.named_parameters()
                if 'classifier' in name or 'head' in name
            }
        else:
            self.adaptation_params = {
                name: param for name, param in base_model.named_parameters()
                if any(layer_name in name for layer_name in adaptation_layers)
            }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """常规前向传播"""
        return self.base_model(x)
    
    def adapt(self, 
              support_x: torch.Tensor, 
              support_y: torch.Tensor,
              adaptation_steps: Optional[int] = None) -> nn.Module:
        """
        基于支持集快速适应
        
        Args:
            support_x: 支持集输入
            support_y: 支持集标签
            adaptation_steps: 适应步数
            
        Returns:
            适应后的模型
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps
        
        # 创建模型副本
        adapted_model = self._copy_model()
        
        # 获取适应参数
        adapted_params = {
            name: param.clone() for name, param in self.adaptation_params.items()
        }
        
        # 内循环：在支持集上优化
        for step in range(adaptation_steps):
            # 前向传播
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss, 
                adapted_params.values(), 
                create_graph=True,
                allow_unused=True
            )
            
            # 更新参数
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.meta_lr * grad
            
            # 更新模型参数
            self._update_model_params(adapted_model, adapted_params)
        
        return adapted_model
    
    def _copy_model(self) -> nn.Module:
        """复制模型"""
        # 这里简化处理，实际应用中需要深度复制
        return self.base_model
    
    def _update_model_params(self, model: nn.Module, params: Dict[str, torch.Tensor]):
        """更新模型参数"""
        for name, param in params.items():
            # 找到对应的模型参数并更新
            module_names = name.split('.')
            current_module = model
            for module_name in module_names[:-1]:
                current_module = getattr(current_module, module_name)
            setattr(current_module, module_names[-1], nn.Parameter(param))


class OnlineLearningWrapper(nn.Module):
    """
    在线学习包装器
    
    支持增量学习和概念漂移适应
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 buffer_size: int = 1000,
                 adaptation_threshold: float = 0.1,
                 learning_rate: float = 0.001):
        
        super().__init__()
        
        self.base_model = base_model
        self.buffer_size = buffer_size
        self.adaptation_threshold = adaptation_threshold
        self.learning_rate = learning_rate
        
        # 经验重放缓冲区
        self.experience_buffer = {
            'x': [],
            'y': [],
            'timestamps': []
        }
        
        # 性能监控
        self.performance_history = []
        self.current_performance = None
        
        # 在线优化器
        self.online_optimizer = torch.optim.Adam(
            self.base_model.parameters(), 
            lr=learning_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.base_model(x)
    
    def update(self, 
               x: torch.Tensor, 
               y: torch.Tensor, 
               timestamp: Optional[float] = None) -> Dict[str, float]:
        """
        在线更新模型
        
        Args:
            x: 新的输入数据
            y: 真实标签
            timestamp: 时间戳
            
        Returns:
            更新信息
        """
        # 评估当前性能
        with torch.no_grad():
            logits = self.base_model(x)
            current_loss = F.cross_entropy(logits, y).item()
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()
        
        # 检测概念漂移
        drift_detected = self._detect_concept_drift(current_loss)
        
        update_info = {
            'loss': current_loss,
            'accuracy': accuracy,
            'drift_detected': drift_detected
        }
        
        # 添加到缓冲区
        self._add_to_buffer(x, y, timestamp)
        
        # 如果检测到漂移，触发模型更新
        if drift_detected or len(self.experience_buffer['x']) % 50 == 0:
            adaptation_loss = self._adapt_model()
            update_info['adaptation_loss'] = adaptation_loss
        
        return update_info
    
    def _detect_concept_drift(self, current_loss: float) -> bool:
        """检测概念漂移"""
        if len(self.performance_history) < 10:
            self.performance_history.append(current_loss)
            return False
        
        # 使用简单的阈值方法
        recent_avg = np.mean(self.performance_history[-10:])
        historical_avg = np.mean(self.performance_history[:-10]) if len(self.performance_history) > 10 else recent_avg
        
        drift_score = abs(current_loss - recent_avg) / (historical_avg + 1e-8)
        
        self.performance_history.append(current_loss)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        return drift_score > self.adaptation_threshold
    
    def _add_to_buffer(self, x: torch.Tensor, y: torch.Tensor, timestamp: Optional[float]):
        """添加经验到缓冲区"""
        self.experience_buffer['x'].append(x.cpu())
        self.experience_buffer['y'].append(y.cpu())
        self.experience_buffer['timestamps'].append(timestamp or len(self.experience_buffer['x']))
        
        # 维护缓冲区大小
        if len(self.experience_buffer['x']) > self.buffer_size:
            self.experience_buffer['x'].pop(0)
            self.experience_buffer['y'].pop(0)
            self.experience_buffer['timestamps'].pop(0)
    
    def _adapt_model(self) -> float:
        """使用缓冲区中的数据适应模型"""