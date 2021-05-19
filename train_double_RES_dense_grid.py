# 3.43
import torch
import numpy as np
import cv2
import argparse

from models import *
from nyu_dataloader import *

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=3, help='size of kernel')
parser.add_argument('--d', type=int, default=15, help='size of grid area')
parser.add_argument('--scale', type=int, default=8, help='scale factor')
parser.add_argument('--parameter',  default='parameter/FDKN_8x', help='name of parameter file')
parser.add_argument('--model',  default='MINE', help='choose model FDKN or DKN')
parser.add_argument('--lr',  default='0.0001', type=float, help='learning rate')
parser.add_argument('--result',  default='./result', help='learning rate')
parser.add_argument('--epoch',  default=30, type=int, help='max epoch')

opt = parser.parse_args()
print(opt)

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-lr_%s-k_%s-d_%s-s_%s'%(opt.result, s, opt.lr, opt.k, opt.d, opt.scale)
if not os.path.exists(result_root): os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)

if opt.model == 'FDKN':
    net = FDKN(kernel_size=opt.k, filter_size=opt.d, residual=True).cuda()
elif opt.model == 'DKN':
    net = DKN(kernel_size=opt.k, filter_size=opt.d, residual=True).cuda()
elif opt.model == 'MINE':
    net1 = MY_FDKN(kernel_size=3, filter_size=opt.d, residual=True).cuda()
    net2 = MY_FDKN(kernel_size=3, filter_size=opt.d, residual=True).cuda()
    
criterion = nn.L1Loss()
optimizer1 = optim.Adam(net1.parameters(), lr=opt.lr)
scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=10000, gamma=0.2)
net1.train()
optimizer2 = optim.Adam(net2.parameters(), lr=opt.lr)
scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=10000, gamma=0.2)
net2.train()

data_transform = transforms.Compose([
    transforms.ToTensor()
])

nyu_dataset = NYU_v2_datset(root_dir='./dataset', scale=opt.scale, transform=data_transform)
dataloader = torch.utils.data.DataLoader(nyu_dataset, batch_size=1, shuffle=True)

def calc_rmse(a, b, minmax):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    a = a*(minmax[1]-minmax[0]) + minmax[1]
    b = b*(minmax[1]-minmax[0]) + minmax[1]
    
    return np.sqrt(np.mean(np.power(a-b,2)))

@torch.no_grad()
def validate(net1, net2, root_dir='./dataset'):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = NYU_v2_datset(root_dir=root_dir, transform=data_transform, train=False)

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    net1.eval()
    net2.eval()
    rmse = np.zeros(449)
    test_minmax = np.load('%s/test_minmax.npy'%root_dir)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        minmax = test_minmax[:,idx]
        
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
        out1 = net1((guidance, target))
        out2 = net2((guidance, out1))
        rmse[idx] = calc_rmse(gt[0,0].cpu().numpy(), out2[0,0].cpu().numpy(), minmax)
        
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse

max_epoch = opt.epoch
for epoch in range(max_epoch):
    net1.train()
    net2.train()
    running_loss = 0.0
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        scheduler1.step()
        scheduler2.step()
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()

        out1 = net1((guidance, target))
        loss1 = criterion(out1, gt)
        loss1.backward()
        optimizer1.step()

        out2 = net2((guidance, out1.detach()))
        loss2 = criterion(out2, gt)
        loss2.backward()    
        optimizer2.step()    

        running_loss += loss2.data.item()
        

        if idx % 50 == 0:
            running_loss /= 50
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch+1, running_loss))
            t.refresh()
            running_loss = 0
            
    rmse = validate(net1,net2)
    logging.info('epoch:%d mean_rmse:%f'%(epoch+1, rmse.mean()))
    torch.save(net1.state_dict(), "%s/parameter%d"%(result_root, epoch+1))