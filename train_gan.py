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
    G = Generator(input_channel=1, kernel_size=3, factor=8).cuda()
    D = Discriminator(input_channel=1, batch_size=opt.batch_size).cuda()


criterion = nn.MSELoss() 
criterion1 = nn.L1Loss() 
optimizer_G = optim.Adam(G.parameters(), lr=opt.lr)
scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=10000, gamma=0.2)
optimizer_D = optim.Adam(D.parameters(), lr=opt.lr)
scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=10000, gamma=0.2)

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
def validate(G, D, root_dir='./dataset'):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = NYU_v2_datset_lr(root_dir=root_dir, transform=data_transform, train=False)

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    G.eval()
    D.eval()

    rmse = np.zeros(449)
    # score_fake = np.zeros(449)
    # score_real = np.zeros(449)

    test_minmax = np.load('%s/test_minmax.npy'%root_dir)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        minmax = test_minmax[:,idx]
        
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
        out = G(target)
        rmse[idx] = calc_rmse(gt[0,0].cpu().numpy(), out[0,0].cpu().numpy(), minmax)

        # score_fake[idk] = D(out).cpu().numpy()
        # score_real[idk] = D(gt).cpu().numpy()
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    # print("score_fake:", score_fake.mean())
    # print("score_real:", score_real.mean())
    return rmse

max_epoch = opt.epoch
best = 1
for epoch in range(max_epoch):
    G.train()
    running_loss = 0.0
    
    real_lable = torch.ones(1).cuda()
    fake_lable = torch.zeros(1).cuda()
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    # rmse = validate(G, D)
    for idx, data in enumerate(t):
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
        
        optimizer_D.zero_grad()

        out = G(target).detach() 
        score_fake = D(out)
        score_real = D(gt)  

        D_loss= criterion(score_real, real_lable) + criterion(score_fake, fake_lable)
        D_loss.backward()  
        optimizer_D.step()
        scheduler_D.step()
        

        optimizer_G.zero_grad()
        out = G(target)  
        score_fake = D(out)
        G_loss_D = criterion(score_fake, real_lable)
        G_loss_MSE = criterion1(gt, out) 
        G_loss = G_loss_MSE + G_loss_D * 0.01
 

        G_loss.backward()  
        
        running_loss += G_loss.data.item()

        optimizer_G.step()
        scheduler_G.step()
        

  
        
        
        if idx % 25 == 0 and idx>0:
            running_loss /= 25
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch+1, running_loss))
            t.refresh()
            running_loss = 0
            
    rmse = validate(G, D)
    logging.info('epoch:%d mean_rmse:%f'%(epoch+1, rmse.mean()))
    torch.save(G.state_dict(), "%s/parameter_G_%d"%(result_root, epoch+1))
    torch.save(D.state_dict(), "%s/parameter_D_%d"%(result_root, epoch+1))
    if rmse.mean()<best:
        best = rmse.mean()
        torch.save(G.state_dict(), "%s/best_G"%(result_root))
        torch.save(D.state_dict(), "%s/best_D"%(result_root))
        best_epoch = epoch+1
    print("best epoch:",best_epoch)
    