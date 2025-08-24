
from torch import nn
import torch
import torch.nn.functional as F
from .attention import SEBlock, ECABlock, CBAMBlock


class DepthwiseSeparableConv2d(nn.Module):
    """
    深度可分离卷积，将标准卷积分解为深度卷积和逐点卷积
    可以减少计算量约8-9倍
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        
        # 深度卷积：每个输入通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        
        # 逐点卷积：1x1卷积，用于调整通道数
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DropBlock2D(nn.Module):
    def __init__(self, drop_rate: float = 0.0, block_size: int = 7):
        super().__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
        gamma = self.drop_rate / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device) < gamma).float()
        mask = F.max_pool2d(mask[:, None, :, :], kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask.squeeze(1)
        mask_sum = mask.sum().clamp(min=1.0)
        norm = mask.numel() / mask_sum
        return x * mask.unsqueeze(1) * norm


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group",
                 activation: str = "swish", attention: str = "se", stride: int = 1,
                 dropout: float = 0.3, dropblock: float = 0.0, 
                 use_depthwise_separable: bool = True):
        super().__init__()
        
        # 选择卷积类型：深度可分离卷积或标准卷积
        if use_depthwise_separable:
            self.conv1 = DepthwiseSeparableConv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.conv2 = DepthwiseSeparableConv2d(out_ch, out_ch, 3, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        
        self.norm1 = self._norm(out_ch, norm)
        self.act1 = self._act(activation)
        self.norm2 = self._norm(out_ch, norm)

        if attention == "se":
            self.attn = SEBlock(out_ch)
        elif attention == "eca":
            self.attn = ECABlock(out_ch)
        elif attention == "cbam":
            self.attn = CBAMBlock(out_ch)
        else:
            self.attn = nn.Identity()

        self.short = nn.Identity() if (in_ch == out_ch and stride == 1) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            self._norm(out_ch, norm)
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.dropblock = DropBlock2D(dropblock) if dropblock > 0 else nn.Identity()
        self.final = self._act(activation)

    def _norm(self, c, t):
        if t == "batch":
            return nn.BatchNorm2d(c)
        if t == "instance":
            return nn.InstanceNorm2d(c)
        if t == "layer":
            return nn.GroupNorm(1, c)
        if t == "group":
            return nn.GroupNorm(min(32, c), c)
        return nn.Identity()

    def _act(self, a):
        if a == "relu":
            return nn.ReLU(inplace=True)
        if a == "swish":
            return nn.SiLU(inplace=True)
        if a == "gelu":
            return nn.GELU()
        if a == "mish":
            return nn.Mish(inplace=True)
        return nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv1(x); x = self.norm1(x); x = self.act1(x); x = self.dropout(x)
        x = self.conv2(x); x = self.norm2(x); x = self.attn(x); x = self.dropblock(x)
        res = self.short(res)
        x = x + res
        return self.final(x)


class OptimizedResidualBlock(nn.Module):
    """
    优化的残差块，使用深度可分离卷积减少计算量
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group",
                 activation: str = "swish", attention: str = "se", stride: int = 1,
                 dropout: float = 0.3, dropblock: float = 0.0):
        super().__init__()
        
        # 第一个卷积：使用深度可分离卷积
        self.conv1 = DepthwiseSeparableConv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = self._norm(out_ch, norm)
        self.act1 = self._act(activation)
        
        # 第二个卷积：也使用深度可分离卷积
        self.conv2 = DepthwiseSeparableConv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = self._norm(out_ch, norm)

        # 注意力机制
        if attention == "se":
            self.attn = SEBlock(out_ch)
        elif attention == "eca":
            self.attn = ECABlock(out_ch)
        elif attention == "cbam":
            self.attn = CBAMBlock(out_ch)
        else:
            self.attn = nn.Identity()

        # 短路连接
        self.short = nn.Identity() if (in_ch == out_ch and stride == 1) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            self._norm(out_ch, norm)
        )
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.dropblock = DropBlock2D(dropblock) if dropblock > 0 else nn.Identity()
        self.final = self._act(activation)

    def _norm(self, c, t):
        if t == "batch":
            return nn.BatchNorm2d(c)
        if t == "instance":
            return nn.InstanceNorm2d(c)
        if t == "layer":
            return nn.GroupNorm(1, c)
        if t == "group":
            return nn.GroupNorm(min(32, c), c)
        return nn.Identity()

    def _act(self, a):
        if a == "relu":
            return nn.ReLU(inplace=True)
        if a == "swish":
            return nn.SiLU(inplace=True)
        if a == "gelu":
            return nn.GELU()
        if a == "mish":
            return nn.Mish(inplace=True)
        return nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv1(x); x = self.norm1(x); x = self.act1(x); x = self.dropout(x)
        x = self.conv2(x); x = self.norm2(x); x = self.attn(x); x = self.dropblock(x)
        res = self.short(res)
        x = x + res
        return self.final(x)
