"""
   frac.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full2Net(torch.nn.Module):
    # 22 hid 18 minimal
    def __init__(self, hid):
        super(Full2Net, self).__init__()
        self.in_hid = torch.nn.Linear(2, hid)
        self.hid_hid2 = torch.nn.Linear(hid, hid)
        self.output = torch.nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.in_hid(input))
        self.hid2 = torch.tanh(self.hid_hid2(self.hid1))
        output = torch.sigmoid(self.output(self.hid2))
        return output

class Full3Net(torch.nn.Module):
    # 20 hid 14 minimal
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.in_hid = torch.nn.Linear(2, hid)
        self.hid_hid2 = torch.nn.Linear(hid, hid)
        self.hid2_hid3 = torch.nn.Linear(hid, hid)
        self.output = torch.nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.in_hid(input))
        self.hid2 = torch.tanh(self.hid_hid2(self.hid1))
        self.hid3 = torch.tanh(self.hid2_hid3(self.hid2))
        output = torch.sigmoid(self.output(self.hid3))
        return output

class DenseNet(torch.nn.Module):
    # 18 hid 12 minimal
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.in_hid = torch.nn.Linear(2, num_hid)
        self.hid_hid2 = torch.nn.Linear(num_hid + 2, num_hid)
        self.output = torch.nn.Linear(2 * num_hid + 2, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.in_hid(input))
        dense1 = torch.cat((self.hid1, input), 1)
        self.hid2 = torch.tanh(self.hid_hid2(dense1))
        dense2 = torch.cat((self.hid2, self.hid1, input), 1)
        output = torch.sigmoid(self.output(dense2))
        return output
