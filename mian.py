import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataLoader import MyDataLoader
import numpy as np
import model

parser = argparse.ArgumentParser(description='MC-cnn')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--datapath', default='processedData.txt',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--loadmodel',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu_device',  default='cuda: 0')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#设置种子
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#初始化model
model=model.MCCNNNet()

#判断是否启用GPU
device = torch.device(args.gpu_device if args.cuda else "cpu")

#加载训练好的model
if args.loadmodel is not None:
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict)


train_datatset = MyDataLoader(args.datapath)
train_loader = torch.utils.data.DataLoader(dataset=train_datatset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

model.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

def main():
    model.train()
    for epoch in range(1, args.epochs):
        i = 0
        for left_image, right_image,label in train_loader:
            left_image = left_image.to(device)
            right_image = right_image.to(device)
            label=label.to(device)
            label= label.float()
            optimizer.zero_grad()
            output = model(left_image,right_image)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            i += 1
            print('[%d][%d] Loss: %.4f' % (epoch, i, loss.item()))

    torch.save(model.state_dict(), args.model_save_path)

if __name__ == '__main__':
   main()
    