import torch
from torch import nn
from .inception_block import InceptionDilatedConvEncoder

class InceptionTSEncoder(nn.Module):
    def __init__(self, input_len, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = InceptionDilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_sizes=[2,5,8]
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        if mask is not None:
            # mask the final timestamp for anomaly detection task
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
            mask &= nan_mask
            x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x

if __name__ == '__main__':
    m1 = InceptionTSEncoder(100, 64, 320, 64, 3, 'binomial')
    w1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f'InceptionTSEncoder: {w1}')