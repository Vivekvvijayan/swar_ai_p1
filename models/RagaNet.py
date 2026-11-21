import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(n_input, n_output, stride=1, kernel_size=80):
    layers = []
    if stride == 1:
        layers.append(nn.Conv1d(n_input, n_output, kernel_size=kernel_size, stride=stride, padding='same'))
    else:
        layers.append(nn.Conv1d(n_input, n_output, kernel_size=kernel_size, stride=stride))
    layers.append(nn.BatchNorm1d(n_output))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, kernel_size=3):
        super().__init__()
        self.conv_block1 = conv_block(n_channels, n_channels, stride=1, kernel_size=kernel_size)
        self.conv_block2 = conv_block(n_channels, n_channels, stride=1, kernel_size=3)

    def forward(self, x):
        identity = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x + identity


class ResNetRagaClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()
        n_input = params.n_input
        n_channel = params.n_channel
        stride = params.stride
        self.n_blocks = params.n_blocks
        self.conv_first = conv_block(n_input, n_channel, stride=stride, kernel_size=80)
        self.max_pool_every = params.max_pool_every

        self.res_blocks = nn.ModuleList([
            ResidualBlock(n_channel, kernel_size=3) for _ in range(self.n_blocks)
        ])

        self.fc1 = nn.Linear(n_channel, params.num_classes)

    def forward(self, x):
        x = self.conv_first(x)

        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if i % self.max_pool_every == 0:
                x = F.max_pool1d(x, 2)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)
