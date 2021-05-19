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
parser.add_argument('--epoch',  default=100, type=int, help='max epoch')
parser.add_argument('--batch_size',  default=16, type=int, help='batch_size')

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
    net = MY_FDKN(kernel_size=3, filter_size=opt.d, residual=True).cuda()
    G = Generator(input_channel=1, kernel_size=3, factor=8).cuda()
    # G.load_state_dict(torch.load('result/gan/best_G'))
    # D = Discriminator(input_channel=1, batch_size=opt.batch_size).cuda()

    
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.2)
net.train()

data_transform = transforms.Compose([
    transforms.ToTensor()
])

nyu_dataset = NYU_v2_datset_lr(root_dir='./dataset', scale=opt.scale, transform=data_transform)
dataloader = torch.utils.data.DataLoader(nyu_dataset, batch_size=opt.batch_size, shuffle=True)

def calc_rmse(a, b, minmax):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    a = a*(minmax[1]-minmax[0]) + minmax[1]
    b = b*(minmax[1]-minmax[0]) + minmax[1]
    
    return np.sqrt(np.mean(np.power(a-b,2)))

@torch.no_grad()
def validate(G, net, root_dir='./dataset'):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = NYU_v2_datset_lr(root_dir=root_dir, transform=data_transform, train=False)

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    net.eval()
    rmse = np.zeros(449)
    test_minmax = np.load('%s/test_minmax.npy'%root_dir)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        minmax = test_minmax[:,idx]
        minmax = (0,1)
        
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
        # out = target
        out_gan = G(target)

        out = net((guidance, out_gan.detach()))
        rmse[idx] = calc_rmse(gt[0,0].cpu().numpy(), out[0,0].cpu().numpy(), minmax)
        
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse

max_epoch = opt.epoch
# rmse = validate(net)
best = 1
for epoch in range(max_epoch):
    net.train()
    running_loss = 0.0
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        optimizer.zero_grad()
        
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()

        with torch.no_grad():
            out_gan = G(target)

        out = net((guidance, out_gan.detach()))

        loss = criterion(out, gt)
        loss.backward()
        
        running_loss += loss.data.item()
        optimizer.step()
        scheduler.step()
        
        if idx % 25 == 0 and idx>0:
            running_loss /= 25
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch+1, running_loss))
            t.refresh()
            running_loss = 0
            
    rmse = validate(G,net)
    logging.info('epoch:%d mean_rmse:%f'%(epoch+1, rmse.mean()))
    torch.save(net.state_dict(), "%s/parameter%d"%(result_root, epoch+1))
    if rmse.mean()<best:
        best = rmse.mean()
        torch.save(net.state_dict(), "%s/best"%(result_root))
        best_epoch = epoch+1
    print("best epoch:",best_epoch)