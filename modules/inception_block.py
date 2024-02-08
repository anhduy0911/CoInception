import torch
from torch import nn
import torch.nn.functional as F
from .modules import WeightNorm
from typing import Union, List

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, weight_norm=False, bias=True, stride=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        if weight_norm:
            self.conv = WeightNorm(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    stride=stride,
                    bias=bias
                ),
                ['weight']
            )
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                stride=stride,
                bias=bias
            )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class BasicUnit(nn.Module):
    def __init__(self, in_channels, out_channels, index, kernel_size):
        super().__init__()
        last_idx = max(0, index - 1)
        base_reception = 2 * kernel_size - 1

        self.last_reception = base_reception**last_idx
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=self.last_reception, weight_norm=False)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=self.last_reception, weight_norm=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)
        return x
    
class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int, index: int,
                 residual: bool, stride: int = 1, hidden_channels: int = 32,
                 kernel_sizes: List[int] = [3, 5], final: bool = False) -> None:
        super().__init__()

        self.final = final
        self.conv_layers = nn.ModuleList([
            BasicUnit(in_channels=in_channels, out_channels=hidden_channels, index=index, kernel_size=kernel_sizes[i])
            for i in range(len(kernel_sizes))
        ])
        self.max_pool_layers = nn.Sequential(*[
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels, hidden_channels, 1),
        ])

        self.residual_fin = nn.Sequential(*[SamePadConv(in_channels=hidden_channels, out_channels=hidden_channels,
                        kernel_size=1, stride=stride, weight_norm=False),
                        # nn.GELU()
        ])

        self.use_residual = residual and in_channels != out_channels
        if self.use_residual:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)

        self.aggregator = nn.Conv1d(hidden_channels*(len(kernel_sizes) + 1), out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, res_prev: Union[List[torch.Tensor], None] = None) -> torch.Tensor:  # type: ignore
        org_x = x
        if res_prev is None:
            res = [conv(x) for conv in self.conv_layers]
        else:
            res = [conv(x) + self.residual_fin(res_prev[i]) for i, conv in enumerate(self.conv_layers)]
        res_mp = self.max_pool_layers(x)
        res.append(res_mp)
        x = torch.cat(res, dim=1)
        x = self.aggregator(x)

        if self.use_residual:
            x = x + self.residual(org_x)
        else:
            x = x + org_x
        return x, res[:-1]

class InceptionDilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_sizes):
        super().__init__()
        self.net = nn.ModuleList([InceptionBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                i+1,
                kernel_sizes=kernel_sizes,
                residual=True,
                hidden_channels=channels[i-1] // 4 if i > 0 else in_channels // 4,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))])
        
    def forward(self, x):
        res_prev = None
        for layer in self.net:
            x, res_prev = layer(x, res_prev)
        return x

