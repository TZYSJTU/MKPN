import torch
import numpy as np
import random
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
parser.add_argument('--model',    default='MINE', help='choose model FDKN or DKN')
parser.add_argument('--k', type=int, default=3, help='size of kpn_kernel')
parser.add_argument('--d', type=int, default=15, help='size of grid area')
parser.add_argument('--result',   default='./result', help='result root')
parser.add_argument('--parameter',  default='parameter/FDKN_8x', help='name of parameter file')
parser.add_argument('--weights',  default='result/fullsize/best', help='root to load model weights')
parser.add_argument('--epoch',    default=3000, type=int, help='max epoch')

parser.add_argument('--scale',      default=8, type=int, help='scale factor')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
parser.add_argument('--num_workers',default=0, type=int, help='num_workers')
parser.add_argument('--lr',         default='0.0001', type=float, help='learning rate')
parser.add_argument('--step_size',  default=100000, type=int, help='step_size decrease lr')
parser.add_argument('--gamma',      default=0.5, type=float, help='gamma decrease lr')
parser.add_argument('--beta',       default=0.1, type=float, help='residual scaling parameter')
parser.add_argument('--residual',   default=False, type=bool, help='output residual')
parser.add_argument('--data_aug',   default=False, type=bool, help='data_augmentation')
parser.add_argument('--load',       default=False, type=bool, help='load model weights')

opt = parser.parse_args()
print(opt)

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-RRDB-baseline'%(opt.result, s)
if not os.path.exists(result_root): os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)
logging.info(
    'scale:%d batch_size:%d lr:%s step_size:%d gamma:%f beta:%f residual:%d data_aug:%d load:%d' \
    %(opt.scale, opt.batch_size, opt.lr, opt.step_size, opt.gamma, opt.beta, opt.residual, opt.data_aug, opt.load)
    )


if opt.model == 'FDKN':
    net = FDKN(kernel_size=opt.k, filter_size=opt.d, residual=True).cuda()
elif opt.model == 'DKN':
    net = DKN(kernel_size=opt.k, filter_size=opt.d, residual=True).cuda()
elif opt.model == 'MINE':
    # net = MY_FDKN(kernel_size=3, filter_size=opt.d, residual=True).cuda()
    net = RRDB_baseline(up_scale=opt.scale, RRDB_channel=64, conv_kernel_size=3, beta=opt.beta, kpn_kernel=opt.k, residual=opt.residual).cuda()
    if opt.load:
        net.load_state_dict(torch.load(opt.weights))
        print("weights loaded")
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
net.train()


######### data division ###########
ID = np.uint8(np.linspace(0,1448,1449))
# random.shuffle(ID)
ID_test = ID[1000:]
ID_train= ID[:1000]

depths = np.load('./dataset/depths.npy')
images = np.load('./dataset/images.npy')

if opt.data_aug:
    Train_dataset = NYU_v2_full_lr_aug(depths=depths[:,:,ID_train],images=images[:,:,:,ID_train], scale=opt.scale, transform=transforms.ToTensor())
else:
    Train_dataset = NYU_v2_full_hr(depths=depths[:,:,ID_train],images=images[:,:,:,ID_train], scale=opt.scale, transform=transforms.ToTensor())

Train_dataloader = torch.utils.data.DataLoader(Train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
Test_dataset = NYU_v2_full_hr(depths=depths[:,:,ID_test],images=images[:,:,:,ID_test], scale=opt.scale, transform=transforms.ToTensor())
Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)

def calc_rmse(a, b, maxmin):
    # a = a[6:-6, 6:-6]
    # b = b[6:-6, 6:-6]
    
    a = a*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()
    b = b*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()
    
    return np.sqrt(np.mean(np.power(a-b,2)))

@torch.no_grad()
def validate(net, dataloader, root_dir='./dataset'):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    net.eval()
    rmse = np.zeros(449)
    # test_minmax = np.load('%s/test_minmax.npy'%root_dir)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        # minmax = test_minmax[:,idx]
        # minmax = (0,1)
        
        guidance, target, gt, maxmin = \
            data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']
        # out = target
        out = net((guidance, target))
        rmse[idx] = calc_rmse(gt[0,0].cpu().numpy(), out[0,0].cpu().numpy(), maxmin)
        
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse

max_epoch = opt.epoch
nan_time = 0
best = 1.0
# rmse = validate(net,Test_dataloader)
for epoch in range(max_epoch):
    net.train()
    running_loss = 0.0
    
    t = tqdm(iter(Train_dataloader), leave=True, total=len(Train_dataloader))
    for idx, data in enumerate(t):
        optimizer.zero_grad()
        
        guidance, target, gt, _ = \
            data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']

        out = net((guidance, target))

        loss = criterion(out, gt)
        running_loss += loss.data.item()
        if np.isnan(running_loss):
            print("nan happened !!!!!!!!!!")
            print(out)
            # net.load_state_dict(torch.load("%s/best"%(result_root)))
            # nan_time += 1    

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if idx % 1 == 0 and idx>0:
            running_loss /= 1
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch, running_loss))
            t.refresh()
            if np.isnan(running_loss):
                print("nan happened !!!!!!!!!!")
                net.load_state_dict(torch.load("%s/best"%(result_root)))
                nan_time += 1      

            running_loss = 0



    if epoch % 1 == 0:      
        rmse = validate(net,Test_dataloader)
        logging.info('epoch:%d mean_rmse:%f'%(epoch, rmse.mean()))
        # torch.save(net.state_dict(), "%s/parameter%d"%(result_root, epoch))
        if rmse.mean()<best:
            best = rmse.mean()
            torch.save(net.state_dict(), "%s/best"%(result_root))
            best_epoch = epoch

        print("best epoch:",best_epoch,"best rmse:",best,"nan happened",nan_time)