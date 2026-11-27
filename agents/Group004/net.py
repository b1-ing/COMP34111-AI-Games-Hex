import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

IN_CHANNELS = 3
N = 11  # default board size

class ResBlock(nn.Module):
    """
    Implements a single residual block.
    Each block consists of:
    Conv1 ->  BN1 -> ReLU -> Conv2 -> BN2 -> Skip connection -> ReLU
    """
    def __init__(self,ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class HexNet(nn.Module):
    """
    Implements the full NN architecture described in alphazero:
    1. Game state (11 x 11 x 3 stack) passed into a 2d convolutional layer which projects it to 64 layers. So 11 x 11 x 64.
    2. Representation is then passed into a sequence of residual blocks as implemented above.
    3. Output of the chain of residual blocks is then passed into a policy head and a value head.
    The policy head feeds the output to 2 1x1 convolutional filters which project the output to 11 x 11 x 2, which is flattened
    to 242 numbers. Then, apply BN -> ReLU -> FC to obtain move probabilities.
    The value head feeds the output to a 1x1 convolutional filter which projects the output to 11 x 11 x 1. BN -> ReLU -> FC is applied to
    convert it to a hidden layer of size 1 x 121. Another ReLU and Fc is applied, before tanh is applied giving a value of [-1,1].
    """
    def __init__(self, n=N, channels=64, blocks=6):
        super().__init__()
        self.n = n
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        # policy head: one logit per cell
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.p_fc = nn.Linear(2 * n * n, n * n)
        # value head: one scalar in [-1, 1]
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
        self.v_fc = nn.Sequential(
            nn.Linear(n * n, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):  # x: (B, C, n, n)
        h = self.body(self.stem(x))
        logits = self.p_fc(self.p_conv(h).flatten(1))  # (B, n*n)
        value = self.v_fc(self.v_conv(h).flatten(1)).squeeze(1)  # (B,)
        return logits, value


