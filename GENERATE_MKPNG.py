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
parser.add_argument('--k', type=int, default=3, help='size of kpn_kernel')
parser.add_argument('--result',   default='./result', help='result root')
parser.add_argument('--weights',  default='./result/MKPN_x/best', help='root to load model weights')
parser.add_argument('--scale',      default=8, type=int, help='scale factor')
parser.add_argument('--num_workers',default=0, type=int, help='num_workers')
parser.add_argument('--channel',    default=16, type=int, help='num_channels')
parser.add_argument('--beta',       default=1, type=float, help='residual scaling parameter')
parser.add_argument('--residual',   default=True, type=bool, help='output residual')
parser.add_argument('--data_aug',   default=False, type=bool, help='data_augmentation')
parser.add_argument('--load',       default=True, type=bool, help='load model weights')

opt = parser.parse_args()
print(opt)

net = MKPNG(up_scale=opt.scale, RRDB_channel=opt.channel, conv_kernel_size=3, beta=opt.beta, kpn_kernel=opt.k, residual=opt.residual).cuda()
if opt.load:
    net.load_state_dict(torch.load(opt.weights))
    print("weights loaded")


# # Sintel 
# Test_dataset = Sintel_full_lr(scale=opt.scale, transform=transforms.ToTensor())
# Test_dataloader = torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False)
# @torch.no_grad()
# def validate_Sintel(net, dataloader):
#     data_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     net.eval()
#     rmse = np.zeros(1064)
    
#     t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
#     for idx, data in enumerate(t):    
#         guidance, target, gt, maxmin = \
#             data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']
        
#         _, _, h, w = target.shape

#         bic = np.array(Image.fromarray(target.squeeze().cpu().numpy()).resize((w*8,h*8),Image.BICUBIC))
#         plt.figure(dpi=206)
#         plt.axis('off')
#         plt.imshow(bic)
#         plt.savefig("./Compare/Bicubic/Sintel/{}_bic.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
#         # plt.show()
#         plt.close('all')

#         gt = gt[0,0].cpu().numpy()
#         plt.figure(dpi=206)
#         plt.axis('off')
#         plt.imshow(gt)
#         plt.savefig("./Compare/GT/Sintel/{}_gt.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
#         # plt.show()
#         plt.close('all')


#         out = net((guidance, target))
#         out = out[0,0].cpu().numpy()
#         out = np.clip(out,0,1)

#         plt.figure(dpi=206)
#         plt.axis('off')
#         plt.imshow(out)
#         plt.savefig("./Compare/MKPNG/Sintel/{}_MKPNG.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
#         # plt.show()
#         plt.close('all')


#         rmse[idx] = calc_rmse(gt, out, maxmin)
#         t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
#         t.refresh()
    
#     return rmse

# if opt.load:
#     rmse = validate_Sintel(net, Test_dataloader)
#     print("rmse:",rmse.mean())


# Middlebury 
Test_dataset = Middlebury_full_lr(scale=opt.scale, transform=transforms.ToTensor())
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
        guidance, target, gt, maxmin = \
            data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']
        
        _, _, h, w = target.shape

        # bic = np.array(Image.fromarray(target.squeeze().cpu().numpy()).resize((w*8,h*8),Image.BICUBIC))
        # plt.figure(dpi=100)
        # plt.axis('off')
        # plt.imshow(bic)
        # plt.savefig("./Compare/Bicubic/Middlebury/{}_bic.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # # plt.show()
        # plt.close('all')

        gt = gt[0,0].cpu().numpy()
        # plt.figure(dpi=100)
        # plt.axis('off')
        # plt.imshow(gt)
        # plt.savefig("./Compare/GT/Middlebury/{}_gt.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # # plt.show()
        # plt.close('all')

        out = net((guidance, target))
        out = out[0,0].cpu().numpy()
        out = np.clip(out,0,1)
        plt.figure(dpi=100)
        plt.axis('off')
        plt.imshow(out)
        plt.savefig("./Compare/MKPNG/Middlebury/{}_MKPNG.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
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
Test_dataset = Lu_full_lr(scale=opt.scale, transform=transforms.ToTensor())
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
        guidance, target, gt, maxmin = \
            data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']
        
        _, _, h, w = target.shape

        # bic = np.array(Image.fromarray(target.squeeze().cpu().numpy()).resize((w*8,h*8),Image.BICUBIC))
        # plt.figure(dpi=130)
        # plt.axis('off')
        # plt.imshow(bic)
        # plt.savefig("./Compare/Bicubic/Lu/{}_bic.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # # plt.show()
        # plt.close('all')

        gt = gt[0,0].cpu().numpy()
        # plt.figure(dpi=130)
        # plt.axis('off')
        # plt.imshow(gt)
        # plt.savefig("./Compare/GT/Lu/{}_gt.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # # plt.show()
        # plt.close('all')

        out = net((guidance, target))
        out = out[0,0].cpu().numpy()
        out = np.clip(out,0,1)
        plt.figure(dpi=130)
        plt.axis('off')
        plt.imshow(out)
        plt.savefig("./Compare/MKPNG/Lu/{}_MKPNG.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
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

Test_dataset = NYU_v2_full_lr(depths=depths[:,:,ID_test],images=images[:,:,:,ID_test], scale=opt.scale, transform=transforms.ToTensor())
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
        
        guidance, target, gt, maxmin = \
            data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['maxmin']

        _, _, h, w = target.shape

        bic = np.array(Image.fromarray(target.squeeze().cpu().numpy()).resize((w*8,h*8),Image.BICUBIC))
        plt.figure(dpi=130)
        plt.axis('off')
        plt.imshow(bic)
        plt.savefig("./Compare/Bicubic/NYU/{}_bic.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')

        gt = gt[0,0].cpu().numpy()
        plt.figure(dpi=130)
        plt.axis('off')
        plt.imshow(gt)
        plt.savefig("./Compare/GT/NYU/{}_gt.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')

        out = net((guidance, target))
        out = out[0,0].cpu().numpy()
        out = np.clip(out,0,1)
        plt.figure(dpi=130)
        plt.axis('off')
        plt.imshow(out)
        plt.savefig("./Compare/MKPNG/NYU/{}_MKPNG.jpg".format(idx), bbox_inches="tight",pad_inches=0.0)
        # plt.show()
        plt.close('all')

        rmse[idx] = calc_rmse_nyu(gt, out, maxmin)
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()

    return rmse

if opt.load:
    rmse = validate_nyu(net, Test_dataloader)
