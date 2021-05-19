from concurrent.futures import ProcessPoolExecutor
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


parser = argparse.ArgumentParser(description="Perform Joint Bilateral Upsampling with a source and reference image")
parser.add_argument('--k', default=3, help='filter size')
parser.add_argument('--sigma1', default=1, help='Sigma of the spatial weights')
parser.add_argument('--sigma2', default=1, help='Sigma of the intensity weights')
parser.add_argument('--scale', default=8, type=int, help='scale factor')
args = parser.parse_args()

mask = torch.zeros((25,25))
mask[0::12,0::12] = 1
mask = mask.view(25**2,1,1)

def generate_Spacial_Kernel(k):

    Kernel = torch.zeros((k,k))
    coodinate = torch.zeros((k,k,2))

    for i in range (k):
        for j in range (k):
            coodinate[i,j,0] = i
            coodinate[i,j,1] = j
    
    central = coodinate[k//2, k//2]

    for i in range (k):
        for j in range (k):
            Kernel[i,j] = torch.sum((coodinate[i,j] - central)**2)
    
    Kernel = torch.exp(-Kernel/(2*args.sigma1**2))

    out = torch.zeros((25,25))
    out[0::12,0::12] = Kernel

    return out
    
    
    
def JBU(depth, image):

    b, c, h, w = image.size()
    image = image[:,0,:,:]*0.299 + image[:,1,:,:]*0.587 * image[:,2,:,:]*0.114
    image = image.unsqueeze(1)

    image_unfold = F.unfold(image, 25, padding=12).view(25**2,h,w)
    depth_unfold = F.unfold(depth, 25, padding=12).view(25**2,h,w)

    Kernel_Spacial = generate_Spacial_Kernel(args.k).view(25**2)

    # Kernel_Spacial = Kernel_Spacial.repeat(1,h,w)

    differ_I = (image[0] - image_unfold)**2
    

    # a = differ_I[:,100,100]
    Kernel_Intensity = torch.exp(-differ_I/(2*args.sigma2**2))*mask

    # a = Kernel_Intensity[:,100,100]

    Kernel = (Kernel_Spacial*Kernel_Intensity.permute(1,2,0)).permute(2,0,1)
    Kernel = Kernel/torch.sum(Kernel,dim=0, keepdim=True)

    # a = Kernel[:,100,100]

    out = torch.sum(Kernel*depth_unfold, dim=0)

    return out


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

        out = JBU(target, guidance)

        # plt.figure(dpi=130)
        # plt.axis('off')
        # plt.imshow(out.numpy())
        # plt.savefig("./Compare/JBU/NYU/{}_bic.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        # plt.close('all')
    
        

        rmse[idx] = calc_rmse_nyu(gt[0,0].numpy(), target[0,0].numpy(), maxmin)  
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()

    return rmse

rmse = validate_nyu(Test_dataloader)





