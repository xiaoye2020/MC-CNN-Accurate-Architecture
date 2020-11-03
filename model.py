from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from submodule import *

class MCCNNNet(nn.Module):
    def __init__(self, maxdisp):
        super(MCCNNNet, self).__init__()
        


    def forward(self, left, right):

        