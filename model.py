from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.optim as optim



def train(imgL,imgR):
        model.train()
        #---------
        # mask = (disp_true > 0)
        # mask.detach_()
        #----

        optimizer.zero_grad()
        output = model(imgL,imgR)
        output = torch.squeeze(output,0)

        test=torch.zero_(1)
        test= Variable(torch.FloatTensor(test))
        loss = nn.BCELoss(output3, dtest, size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data[0]


class MCCNNNet(nn.Module):
    def __init__(self):
        super(MCCNNNet, self).__init__()
        self.feature_extraction=feature_extraction()
        self.layer1 = nn.Sequential(nn.Linear(9*9*112*2,384), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(384,384), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(384,384), nn.ReLU(inplace=True))
        self.layer4 = nn.Linear(384,1)


    def forward(self, left, right):
        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)
        left_linear=refimg_fea.view([refimg_fea.size()[0],-1])
        right_linear=targetimg_fea.view([refimg_fea.size()[0],-1])

        linear_input=torch.cat([left_linear,right_linear],dim=1)
        output=self.layer1(linear_input)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=F.sigmoid(output)
        return output


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=112,kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=112,out_channels=112,kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=112,out_channels=112,kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=112,out_channels=112,kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True))
       

    def forward(self, image):
        output=self.layer1(image)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        
        return output
  

model=MCCNNNet()

imgL=torch.rand(1,1,9, 9)
imgL= Variable(torch.FloatTensor(imgL))

imgR=torch.rand(1,1,9, 9)
imgR= Variable(torch.FloatTensor(imgR))

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

train(imgL,imgR)