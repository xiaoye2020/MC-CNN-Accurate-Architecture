import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import model

parser = argparse.ArgumentParser(description='MC-cnn')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=0,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= './trained/pretrained_sceneflow.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#设置种子
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#初始化model
model=model.MCCNNNet(args.maxdisp)

#判断是否启用GPU
if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

#加载训练好的model
if args.loadmodel is not None:
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict)

optimizer = optim.SGD(model.parameters(), lr=0.003, betas=(0.9, 0.999))










def main():




if __name__ == '__main__':
   main()
    