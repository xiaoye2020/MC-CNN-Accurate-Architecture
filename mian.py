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

def train(imgL,imgR,disp_L):
        model.train()
        #---------
        mask = (disp_true > 0)
        mask.detach_()
        #----

        optimizer.zero_grad()
        output = model(imgL,imgR)
        output = torch.squeeze(output,1)
        loss = nn.BCELoss(output[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data[0]

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR)

        pred_disp = output3.data.cpu()

        #computing 3-px error#
        true_disp = disp_true
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()

        return 1-(float(torch.sum(correct))/float(len(index[0])))

def main():
	max_acc=0
	max_epo=0
	start_full_time = time.time()

	for epoch in range(1, args.epochs+1):
	   total_train_loss = 0
	   total_test_loss = 0
	   adjust_learning_rate(optimizer,epoch)
           
               ## training ##
           for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
               start_time = time.time() 

               loss = train(imgL_crop,imgR_crop, disp_crop_L)
	       print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
	       total_train_loss += loss
	   print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
	   
               ## Test ##

           for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
               test_loss = test(imgL,imgR, disp_L)
               print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
               total_test_loss += test_loss


	   print('epoch %d total 3-px error in val = %.3f' %(epoch, total_test_loss/len(TestImgLoader)*100))
	   if total_test_loss/len(TestImgLoader)*100 > max_acc:
		max_acc = total_test_loss/len(TestImgLoader)*100
		max_epo = epoch
	   print('MAX epoch %d total test error = %.3f' %(max_epo, max_acc))

	   #SAVE
	   savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
	   torch.save({
		    'epoch': epoch,
		    'state_dict': model.state_dict(),
		    'train_loss': total_train_loss/len(TrainImgLoader),
		    'test_loss': total_test_loss/len(TestImgLoader)*100,
		}, savefilename)
	
        print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
	print(max_epo)
	print(max_acc)


if __name__ == '__main__':
   main()
    