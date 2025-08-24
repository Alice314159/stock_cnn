
from torch import nn
from .blocks import ResidualBlock

class StockCNN(nn.Module):
    """Optimized 2D CNN for (C=1, T, F) inputs with residual blocks and attention."""
    def __init__(self, num_classes: int = 2, base_channels: int = 16, channel_multiplier: float = 1.5,
                 num_blocks: int = 2, dropout: float = 0.3, norm: str = "group",
                 activation: str = "swish", attention: str = "se"):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, base_channels), base_channels),
            nn.SiLU(inplace=True) if activation == "swish" else nn.ReLU(inplace=True)
        )

        blocks = []
        in_ch = base_channels
        for i in range(num_blocks):
            out_ch = int(base_channels * (channel_multiplier ** i))
            stride = 2 if i > 0 else 1
            blocks.append(ResidualBlock(in_ch, out_ch, norm=norm, activation=activation,
                                        attention=attention, stride=stride, dropout=dropout, dropblock=0.0))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, in_ch // 2),
            nn.SiLU(inplace=True) if activation == "swish" else nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_ch // 2, num_classes),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x
