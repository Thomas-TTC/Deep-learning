"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.log_softmax(self.linear(x), 1)
        return x

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.hidden = nn.Linear(28 * 28, 120)
        self.output = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.tanh(self.hidden(x))
        x = torch.log_softmax(self.output(x), 1)
        return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE2
        self.conv1 = nn.Conv2d(1,   14, 5, padding=2)
        self.conv2 = nn.Conv2d(14, 40, 5)
        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.linear = nn.Linear(40 * 5 * 5, 200)
        self.output = nn.Linear(200, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxPool(x)
        x = torch.relu(self.conv2(x))
        x = self.maxPool(x)
        x = x.view(-1, 40 * 5 * 5)
        x = torch.relu(self.linear(x))
        x = torch.log_softmax(self.output(x), 0)
        return x
