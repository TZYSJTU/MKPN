import torch
import numpy as np
import random
import cv2
import argparse

from models import *
from nyu_dataloader import *

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms, utils

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os

def calc_rmse(a, b, maxmin):
    
    a = a*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()
    b = b*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()

    a = a.astype(np.uint8)
    b = b.astype(np.uint8)
    return np.sqrt(np.mean(np.power(a-b,2)))

def calc_rmse_nyu(a, b, maxmin):
    
    a = a*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()
    b = b*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()

    return np.sqrt(np.mean(np.power(a-b,2)))

parser = argparse.ArgumentParser()
parser.add_argument('--result',   default='./result', help='result root')
parser.add_argument('--weights',  default='./result/DMSG_8x/best', help='root to load model weights')
parser.add_argument('--epoch',    default=3000, type=int, help='max epoch')
parser.add_argument('--scale',      default=8, type=int, help='scale factor')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
parser.add_argument('--num_workers',default=0, type=int, help='num_workers')
parser.add_argument('--lr',         default='0.0001', type=float, help='learning rate')
parser.add_argument('--step_size',  default=50000, type=int, help='step_size decrease lr')
parser.add_argument('--gamma',      default=0.8, type=float, help='gamma decrease lr')
parser.add_argument('--residual',   default=True, type=bool, help='output residual')
parser.add_argument('--data_aug',   default=False, type=bool, help='data_augmentation')
parser.add_argument('--load',       default=True, type=bool, help='load model weights')

opt = parser.parse_args()
print(opt)

net = DMSG().cuda()
if opt.load:
    net.load_state_dict(torch.load(opt.weights))
    print("weights loaded")

# Sintel 
Test_dataset = Sintel_full_hr_lr(scale=opt.scale, transform=transforms.ToTensor())
Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)
@torch.no_grad()
def validate_Sintel(net, dataloader):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    net.eval()
    rmse = np.zeros(1064)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):    
        guidance, lr, target, gt, maxmin = \
            data['guidance'].cuda(), data['lr'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']
      
        out = net((guidance, lr, target))
        
        _, _, h, w = target.shape
        gt = gt[0,0].cpu().numpy()

        out = out[0,0].cpu().numpy()
        out = np.clip(out,0,1)
        

        # plt.figure(dpi=206)
        # plt.axis('off')
        # plt.imshow(out)
        # plt.savefig("./Compare/DMSG/Sintel/{}_DMSG.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # # plt.show()
        # plt.close('all')

        rmse[idx] = calc_rmse(gt, out, maxmin)
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse

if opt.load:
    rmse = validate_Sintel(net, Test_dataloader)
    print("rmse:",rmse.mean())


# Middlebury 
Test_dataset = Middlebury_full_hr_lr(scale=opt.scale, transform=transforms.ToTensor())
Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)
@torch.no_grad()
def validate_middlebury(net, dataloader):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    net.eval()
    rmse = np.zeros(30)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):    
        guidance, lr, target, gt, maxmin = \
            data['guidance'].cuda(), data['lr'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']
      
        out = net((guidance, lr, target))
        
        _, _, h, w = target.shape
        gt = gt[0,0].cpu().numpy()

        out = out[0,0].cpu().numpy()
        out = np.clip(out,0,1)
        

        plt.figure(dpi=100)
        plt.axis('off')
        plt.imshow(out)
        plt.savefig("./Compare/DMSG/Middlebury/{}_DMSG.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')

        rmse[idx] = calc_rmse(gt, out, maxmin)
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse

if opt.load:
    rmse = validate_middlebury(net, Test_dataloader)
    print("rmse:",rmse.mean())

# Lu 
Test_dataset = Lu_full_hr_lr(scale=opt.scale, transform=transforms.ToTensor())
Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)
@torch.no_grad()
def validate_Lu(net, dataloader):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    net.eval()
    rmse = np.zeros(6)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):    
        guidance, lr, target, gt, maxmin = \
            data['guidance'].cuda(), data['lr'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']
      
        out = net((guidance, lr, target))
        
        _, _, h, w = target.shape
        gt = gt[0,0].cpu().numpy()

        out = out[0,0].cpu().numpy()
        out = np.clip(out,0,1)
        

        plt.figure(dpi=130)
        plt.axis('off')
        plt.imshow(out)
        plt.savefig("./Compare/DMSG/Lu/{}_DMSG.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')

        rmse[idx] = calc_rmse(gt, out, maxmin)
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse

if opt.load:
    rmse = validate_Lu(net, Test_dataloader)
    print("rmse:",rmse.mean())


# ######### nyu v2 division ###########
ID = np.uint8(np.linspace(0,1448,1449))
# random.shuffle(ID)
ID_test = ID[1000:]

depths = np.load('./dataset/depths.npy')
images = np.load('./dataset/images.npy')

Test_dataset = NYU_v2_full_hr_lr(depths=depths[:,:,ID_test],images=images[:,:,:,ID_test], scale=opt.scale, transform=transforms.ToTensor())
Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)



@torch.no_grad()
def validate_nyu(net, dataloader,):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    net.eval()
    rmse = np.zeros(449)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        # minmax = test_minmax[:,idx]
        # minmax = (0,1)
        
        guidance, lr, target, gt, maxmin = \
            data['guidance'].cuda(), data['lr'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']
      
        out = net((guidance, lr, target))
        
        _, _, h, w = target.shape
        gt = gt[0,0].cpu().numpy()

        out = out[0,0].cpu().numpy()
        out = np.clip(out,0,1)
        
        plt.figure(dpi=130)
        plt.axis('off')
        plt.imshow(out)
        plt.savefig("./Compare/DMSG/NYU/{}_DMSG.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')

        rmse[idx] = calc_rmse_nyu(gt, out, maxmin)
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()

    return rmse

if opt.load:
    rmse = validate_nyu(net, Test_dataloader)
