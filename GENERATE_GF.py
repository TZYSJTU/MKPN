from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np
import random
import cv2
import argparse

from box_filter import *
from guided_filter import *
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


parser = argparse.ArgumentParser(description="Perform Joint Bilateral Upsampling with a source and reference image")
parser.add_argument('--r', default=2, help='filter r')
parser.add_argument('--sigma1', default=1, help='Sigma of the spatial weights')
parser.add_argument('--scale', default=8, type=int, help='scale factor')
args = parser.parse_args()
    
Filter = GuidedFilter(args.r,eps=0.00001)    
def GF(depth, image):
    image = image[:,0,:,:]*0.299 + image[:,1,:,:]*0.587 * image[:,2,:,:]*0.114
    image = image.unsqueeze(1)

    out = Filter(image, depth)

    return out


def calc_rmse(a, b, maxmin):
    
    a = a*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()
    b = b*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()

    a = a.astype(np.uint8)
    b = b.astype(np.uint8)
    return np.sqrt(np.mean(np.power(a-b,2)))

def calc_rmse_nyu(a, b, maxmin):
    # a = a[6:-6, 6:-6]
    # b = b[6:-6, 6:-6]
    
    a = a*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()
    b = b*(maxmin[0].cpu().numpy()-maxmin[1].cpu().numpy()) + maxmin[1].cpu().numpy()

    return np.sqrt(np.mean(np.power(a-b,2)))

# Sintel 
Test_dataset = Sintel_full_hr(scale=args.scale, transform=transforms.ToTensor())
Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)
@torch.no_grad()
def validate_Sintel(dataloader):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    rmse = np.zeros(1064)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):    
        guidance, target, gt, maxmin = data['guidance'], data['target'], data['gt'], data['maxmin']

        out = GF(target, guidance)

        plt.figure(dpi=206)
        plt.axis('off')
        plt.imshow(out[0,0].numpy())
        plt.savefig("./Compare/GF/Sintel/{}_gf.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')
    
        

        rmse[idx] = calc_rmse_nyu(gt[0,0].numpy(), out[0,0].numpy(), maxmin)  
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse


rmse = validate_Sintel(Test_dataloader)
print("rmse:",rmse.mean())


# Middlebury 
Test_dataset = Middlebury_full_hr(scale=args.scale, transform=transforms.ToTensor())
Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)
@torch.no_grad()
def validate_middlebury(dataloader):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    rmse = np.zeros(30)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):    
        guidance, target, gt, maxmin = data['guidance'], data['target'], data['gt'], data['maxmin']

        out = GF(target, guidance)

        plt.figure(dpi=100)
        plt.axis('off')
        plt.imshow(out[0,0].numpy())
        plt.savefig("./Compare/GF/Middlebury/{}_gf.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')
    
        

        rmse[idx] = calc_rmse_nyu(gt[0,0].numpy(), out[0,0].numpy(), maxmin)  
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse

rmse = validate_middlebury(Test_dataloader)
print("rmse:",rmse.mean())

# Lu 
Test_dataset = Lu_full_hr(scale=args.scale, transform=transforms.ToTensor())
Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)
@torch.no_grad()
def validate_Lu(dataloader):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    rmse = np.zeros(6)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):    
        guidance, target, gt, maxmin = data['guidance'], data['target'], data['gt'], data['maxmin']

        out = GF(target, guidance)

        plt.figure(dpi=130)
        plt.axis('off')
        plt.imshow(out[0,0].numpy())
        plt.savefig("./Compare/GF/Lu/{}_gf.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')
    
        

        rmse[idx] = calc_rmse_nyu(gt[0,0].numpy(), out[0,0].numpy(), maxmin)  
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse


rmse = validate_Lu(Test_dataloader)
print("rmse:",rmse.mean())



# ######### nyu v2 division ###########
ID = np.uint8(np.linspace(0,1448,1449))
# random.shuffle(ID)
ID_test = ID[1000:]

depths = np.load('./dataset/depths.npy')
images = np.load('./dataset/images.npy')

print("data loaded")
Test_dataset = NYU_v2_full_hr(depths=depths[:,:,ID_test],images=images[:,:,:,ID_test], scale=args.scale, transform=transforms.ToTensor())
Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)

@torch.no_grad()
def validate_nyu(dataloader):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    rmse = np.zeros(449)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        guidance, target, gt, maxmin = data['guidance'], data['target'], data['gt'], data['maxmin']

        # target = target[0,0].numpy()
        # guidance = (guidance[0].permute(1,2,0).numpy()*255).astype(np.uint8)
        # gt = gt[0,0].numpy()

        out = GF(target, guidance)

        plt.figure(dpi=130)
        plt.axis('off')
        plt.imshow(out[0,0].numpy())
        plt.savefig("./Compare/GF/NYU/{}_gf.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')
    
        

        rmse[idx] = calc_rmse_nyu(gt[0,0].numpy(), out[0,0].numpy(), maxmin)  
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()

    return rmse

rmse = validate_nyu(Test_dataloader)





