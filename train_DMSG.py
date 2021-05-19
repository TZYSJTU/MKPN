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
parser.add_argument('--result',   default='./result', help='result root')
parser.add_argument('--weights',  default='result/best', help='root to load model weights')
parser.add_argument('--epoch',    default=3000, type=int, help='max epoch')

parser.add_argument('--scale',      default=8, type=int, help='scale factor')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
parser.add_argument('--num_workers',default=0, type=int, help='num_workers')
parser.add_argument('--lr',         default='0.0001', type=float, help='learning rate')
parser.add_argument('--step_size',  default=100000, type=int, help='step_size decrease lr')
parser.add_argument('--gamma',      default=0.8, type=float, help='gamma decrease lr')
parser.add_argument('--data_aug',   default=False, type=bool, help='data_augmentation')
parser.add_argument('--load',       default=False, type=bool, help='load model weights')

opt = parser.parse_args()
print(opt)

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/DMSG_%s-s_%s'%(opt.result, s, opt.scale)
if not os.path.exists(result_root): os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)

# net = MY_FDKN(kernel_size=3, filter_size=opt.d, residual=True).cuda()
net = DMSG().cuda()
if opt.load:
    net.load_state_dict(torch.load(opt.weights))
    print("weights loaded")

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

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
    Train_dataset = NYU_v2_full_hr_lr(depths=depths[:,:,ID_train],images=images[:,:,:,ID_train], scale=opt.scale, transform=transforms.ToTensor())
print("data loaded")

Train_dataloader = torch.utils.data.DataLoader(Train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
Test_dataset = NYU_v2_full_hr_lr(depths=depths[:,:,ID_test],images=images[:,:,:,ID_test], scale=opt.scale, transform=transforms.ToTensor())
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
        
        guidance, lr, target, gt, maxmin = \
            data['guidance'].cuda(), data['lr'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']
        # out = target
        out = net((guidance, lr, target))
        rmse[idx] = calc_rmse(gt[0,0].cpu().numpy(), out[0,0].cpu().numpy(), maxmin)
        
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse

max_epoch = opt.epoch
best = 1.0
if opt.load:
    rmse = validate(net,Test_dataloader)
for epoch in range(max_epoch):
    net.train()
    running_loss = 0.0
    
    t = tqdm(iter(Train_dataloader), leave=True, total=len(Train_dataloader))
    for idx, data in enumerate(t):
        optimizer.zero_grad()
        
        guidance, lr, target, gt, maxmin = \
            data['guidance'].cuda(), data['lr'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']

        out = net((guidance, lr, target))

        loss = criterion(out, gt)
        running_loss += loss.data.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if idx % 1 == 0 and idx>0:
            running_loss /= 1
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch, running_loss))
            t.refresh()    

            running_loss = 0



    if epoch % 2 == 0:      
        rmse = validate(net,Test_dataloader)
        logging.info('epoch:%d mean_rmse:%f'%(epoch, rmse.mean()))
        # torch.save(net.state_dict(), "%s/parameter%d"%(result_root, epoch))
        if rmse.mean()<best:
            best = rmse.mean()
            torch.save(net.state_dict(), "%s/best"%(result_root))
            best_epoch = epoch

        print("best epoch:",best_epoch,"best rmse:",best,"nan happened")