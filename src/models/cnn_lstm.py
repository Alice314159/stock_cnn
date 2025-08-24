import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .blocks import ResidualBlock, OptimizedResidualBlock
from .attention import SEBlock, ECABlock, CBAMBlock

class ImprovedCNNLSTMModel(nn.Module):
    """
    改进的CNN-LSTM混合模型
    
    主要改进：
    1. 更好的CNN-LSTM连接方式
    2. 时间注意力机制
    3. 多尺度特征提取
    4. 残差连接
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 base_channels: int = 16,
                 channel_multiplier: float = 1.5,
                 num_blocks: int = 2,
                 dropout: float = 0.3,
                 norm: str = "group",
                 activation: str = "swish",
                 attention: str = "se",
                 
                 # LSTM配置
                 use_lstm: bool = True,
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 lstm_bidirectional: bool = True,
                 
                 # 新增配置
                 use_temporal_attention: bool = True,
                 use_multiscale: bool = True,
                 temporal_scales: List[int] = [3, 5, 7],  # 不同的时间尺度
                 
                 # 输入配置
                 window_size: int = 60,
                 num_features: int = 20):
        
        super().__init__()
        
        self.use_lstm = use_lstm
        self.window_size = window_size
        self.num_features = num_features
        self.use_temporal_attention = use_temporal_attention
        self.use_multiscale = use_multiscale
        
        if use_multiscale:
            # 多尺度CNN提取器
            self.multi_scale_cnns = nn.ModuleList([
                self._build_single_scale_cnn(
                    base_channels, channel_multiplier, num_blocks,
                    dropout, norm, activation, attention, scale
                ) for scale in temporal_scales
            ])
            # 计算总的特征维度
            single_cnn_output = self._calculate_cnn_output_size(base_channels, channel_multiplier, num_blocks)
            total_cnn_output = single_cnn_output * len(temporal_scales)
        else:
            # 单尺度CNN
            self.cnn = self._build_single_scale_cnn(
                base_channels, channel_multiplier, num_blocks,
                dropout, norm, activation, attention
            )
            total_cnn_output = self._calculate_cnn_output_size(base_channels, channel_multiplier, num_blocks)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_cnn_output, total_cnn_output // 2),
            self._get_activation(activation),
            nn.Dropout(dropout)
        )
        fusion_output_size = total_cnn_output // 2
        
        if use_lstm:
            # 改进的时序建模方法
            # 方法1：将CNN特征重新排列为时序
            self.temporal_projection = nn.Linear(fusion_output_size, lstm_hidden_size)
            
            # LSTM层
            self.lstm = nn.LSTM(
                input_size=lstm_hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout if lstm_num_layers > 1 else 0,
                bidirectional=lstm_bidirectional,
                batch_first=True
            )
            
            lstm_output_size = lstm_hidden_size * (2 if lstm_bidirectional else 1)
            
            # 时间注意力机制
            if use_temporal_attention:
                self.temporal_attention = TemporalAttention(lstm_output_size)
            
            # 分类头
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_size // 2, num_classes)
            )
        else:
            # 仅CNN的分类头
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fusion_output_size, fusion_output_size // 2),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(fusion_output_size // 2, num_classes)
            )
    
    def _build_single_scale_cnn(self, base_channels: int, channel_multiplier: float,
                               num_blocks: int, dropout: float, norm: str,
                               activation: str, attention: str, scale: int = None) -> nn.Module:
        """构建单尺度CNN"""
        
        # 如果指定了尺度，调整卷积核大小
        kernel_size = scale if scale is not None else 3
        padding = kernel_size // 2
        
        # 输入层
        input_layer = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size, padding=padding, bias=False),
            self._get_norm(norm, base_channels),
            self._get_activation(activation)
        )
        
        # 残差块
        blocks = []
        in_ch = base_channels
        for i in range(num_blocks):
            out_ch = int(base_channels * (channel_multiplier ** i))
            stride = 2 if i > 0 else 1
            
            blocks.append(OptimizedResidualBlock(
                in_ch, out_ch,
                norm=norm,
                activation=activation,
                attention=attention,
                stride=stride,
                dropout=dropout,
                dropblock=0.0
            ))
            in_ch = out_ch
        
        # 全局平均池化
        pool = nn.AdaptiveAvgPool2d(1)
        
        return nn.Sequential(input_layer, *blocks, pool)
    
    def _calculate_cnn_output_size(self, base_channels: int, 
                                  channel_multiplier: float, 
                                  num_blocks: int) -> int:
        """计算CNN输出特征维度"""
        output_channels = base_channels
        for i in range(num_blocks):
            output_channels = int(base_channels * (channel_multiplier ** i))
        return output_channels
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == "swish":
            return nn.SiLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "mish":
            return nn.Mish(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def _get_norm(self, norm: str, num_channels: int) -> nn.Module:
        """获取归一化层"""
        if norm == "batch":
            return nn.BatchNorm2d(num_channels)
        elif norm == "instance":
            return nn.InstanceNorm2d(num_channels)
        elif norm == "layer":
            return nn.LayerNorm([num_channels, 1, 1])
        else:  # group
            return nn.GroupNorm(min(32, num_channels), num_channels)
    
    def forward(self, x: torch.Tensor, stock_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps, features)
            stock_indices: (batch_size,) 股票索引
        """
        # 输入数据验证
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("输入数据包含NaN或Inf值")
        
        batch_size = x.size(0)
        
        if self.use_multiscale:
            # 多尺度特征提取
            multiscale_features = []
            for cnn in self.multi_scale_cnns:
                features = cnn(x)  # (batch_size, channels, 1, 1)
                features = features.squeeze(-1).squeeze(-1)
                # 检查特征是否有效
                if torch.isnan(features).any() or torch.isinf(features).any():
                    raise ValueError("CNN特征提取产生NaN或Inf值")
                multiscale_features.append(features)
            
            # 融合多尺度特征
            cnn_features = torch.cat(multiscale_features, dim=1)
        else:
            # 单尺度特征提取
            cnn_features = self.cnn(x)
            cnn_features = cnn_features.squeeze(-1).squeeze(-1)
            # 检查特征是否有效
            if torch.isnan(cnn_features).any() or torch.isinf(cnn_features).any():
                raise ValueError("CNN特征提取产生NaN或Inf值")
        
        # 特征融合
        fused_features = self.feature_fusion(cnn_features)
        # 检查融合特征是否有效
        if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
            raise ValueError("特征融合产生NaN或Inf值")
        
        if self.use_lstm:
            # 改进的时序建模方法
            # 方法：使用滑动窗口创建时序
            # 将融合后的特征投影到LSTM输入空间
            projected_features = self.temporal_projection(fused_features)
            # 检查投影特征是否有效
            if torch.isnan(projected_features).any() or torch.isinf(projected_features).any():
                raise ValueError("特征投影产生NaN或Inf值")
            
            # 创建简单的时序：重复特征作为多个时间步
            # 在实际应用中，可以考虑更复杂的时序重构方法
            seq_length = 8  # 可配置
            lstm_input = projected_features.unsqueeze(1).repeat(1, seq_length, 1)
            
            # LSTM处理
            lstm_out, _ = self.lstm(lstm_input)  # (batch_size, seq_length, hidden_size)
            # 检查LSTM输出是否有效
            if torch.isnan(lstm_out).any() or torch.isinf(lstm_out).any():
                raise ValueError("LSTM输出产生NaN或Inf值")
            
            # 时间注意力
            if self.use_temporal_attention:
                lstm_out = self.temporal_attention(lstm_out)  # (batch_size, hidden_size)
            else:
                # 使用最后一个时间步的输出
                lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
            
            # 分类
            output = self.classifier(lstm_out)
        else:
            # 仅使用CNN特征
            output = self.classifier(fused_features)
        
        # 最终输出验证
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError("模型输出产生NaN或Inf值")
        
        return output


class TemporalAttention(nn.Module):
    """时间注意力机制"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch_size, seq_length, hidden_size)
        Returns:
            weighted_output: (batch_size, hidden_size)
        """
        # 计算注意力权重
        attention_weights = self.attention(lstm_output)  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # 在序列维度上归一化
        
        # 加权求和
        weighted_output = torch.sum(lstm_output * attention_weights, dim=1)  # (batch_size, hidden_size)
        
        return weighted_output


class EfficientMultiStockModel(nn.Module):
    """
    高效的多股票模型
    
    改进：
    1. 共享的特征提取器 + 股票特定的头部
    2. 批量处理而非逐个处理
    3. 股票嵌入
    """
    
    def __init__(self, 
                 num_stocks: int,
                 num_classes: int = 2,
                 embedding_dim: int = 32,
                 **model_kwargs):
        
        super().__init__()
        
        self.num_stocks = num_stocks
        self.embedding_dim = embedding_dim
        
        # 股票嵌入
        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim)
        
        # 共享的特征提取器
        self.shared_extractor = ImprovedCNNLSTMModel(
            num_classes=128,  # 中间特征维度
            **model_kwargs
        )
        
        # 修改共享提取器的分类头为特征输出
        feature_size = 128
        self.shared_extractor.classifier = nn.Sequential(
            nn.Dropout(model_kwargs.get('dropout', 0.3)),
            nn.Linear(
                self.shared_extractor.classifier[1].in_features,
                feature_size
            ),
            nn.ReLU(),
            nn.Dropout(model_kwargs.get('dropout', 0.3))
        )
        
        # 最终分类器
        self.final_classifier = nn.Sequential(
            nn.Linear(feature_size + embedding_dim, feature_size),
            nn.ReLU(),
            nn.Dropout(model_kwargs.get('dropout', 0.3)),
            nn.Linear(feature_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor, stock_indices: torch.Tensor) -> torch.Tensor:
        """
        高效的前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, window_size, num_features)
            stock_indices: 股票索引 (batch_size,)
            
        Returns:
            输出张量 (batch_size, num_classes)
        """
        # 批量处理所有样本
        shared_features = self.shared_extractor(x)  # (batch_size, feature_size)
        
        # 获取股票嵌入
        stock_embeds = self.stock_embedding(stock_indices)  # (batch_size, embedding_dim)
        
        # 融合共享特征和股票特定信息
        combined_features = torch.cat([shared_features, stock_embeds], dim=1)
        
        # 最终预测
        output = self.final_classifier(combined_features)
        
        return output