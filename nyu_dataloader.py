import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils



class NYU_v2_datset(Dataset):
    """NYUDataset.
    for DKN and FDKN 
    """

    def __init__(self, root_dir, scale=8, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            self.images = np.load('%s/train_images_split.npy'%root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            self.images = np.load('%s/test_images_v2.npy'%root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        
        h, w = depth.shape
        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w, h), Image.BICUBIC))
        # print(depth[:4,:8])
        # print(target[:4,:8])

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth}
        
        return sample
    
        """
        return:
            sample:
            guidance (np.array float): H x W x 3 
            target ((np.array float)): H x W x 1
            gt ((np.array float)): H x W x 1
            
        """



class NYU_v2_datset_lr(Dataset):
    """
        NYU v2 Dataset.
        halfsize 480 x 320
        input is lr 480/s x 320/s 
    """
    def __init__(self, root_dir, scale=8, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            self.images = np.load('%s/train_images_split.npy'%root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            self.images = np.load('%s/test_images_v2.npy'%root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        
        h, w = depth.shape
        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))
        # print(depth[:4,:8])
        # print(target[:4,:8])

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth}
        
        return sample
    
        """
        return:
            sample:
            guidance (np.array float): H x W x 3 
            target ((np.array float)): H x W x 1
            gt ((np.array float)): H x W x 1
            
        """


class NYU_v2_full_lr(Dataset):
    """
        NYU v2 Dataset.
        fullsize 480 x 640
        input is lr 480/s x 640/s 
        No augmentation
    """
    def __init__(self, depths, images, scale=8, transform=None):
        self.scale = scale
        self.transform = transform
        
        self.depths = depths
        self.images = images

    def __len__(self):
        return self.depths.shape[2]

    def __getitem__(self, idx):
        depth = self.depths[:,:,idx]
        image = self.images[:,:,:,idx]
        Max = depth.max()
        Min = depth.min()  
        depth = (depth-Min)/(Max-Min)

        h, w = depth.shape

        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))
        # print(depth[:4,:8])
        # print(target[:4,:8])

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(Max,Min)}
        
        return sample

class NYU_v2_full_lr_aug(Dataset):
    """
        NYU v2 Dataset.
        fullsize 480 x 640
        input is lr 480/s x 640/s 
        augmentation
    """
    def __init__(self, depths, images, scale=8, transform=None):
        self.depths = depths
        self.images = images
        self.scale = scale
        self.transform = transform
        
        self.HorizontalFlip = transforms.RandomHorizontalFlip(p=1)
        self.VerticalFlip = transforms.RandomVerticalFlip(p=1)
        self.rotate = transforms.RandomRotation(180)


    def __len__(self):
        return self.depths.shape[2]

    def __getitem__(self, idx):
        depth = self.depths[:,:,idx]
        image = self.images[:,:,:,idx]
        # image = np.float32(image)
        # image /= 255
        # image = np.transpose(image,(2,0,1))
        Max = depth.max()
        Min = depth.min()  
        depth = (depth-Min)/(Max-Min)

        h, w = depth.shape
        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))
        # image = Image.fromarray(image)
        # print(depth[:4,:8])
        # print(target[:4,:8])
        image = Image.fromarray(image)
        depth = Image.fromarray(depth)
        target = Image.fromarray(target)  

        p = random.randint(1,2)
        if p == 1:
            image = self.HorizontalFlip(image)
            depth = self.HorizontalFlip(depth)
            target = self.HorizontalFlip(target)          
        # elif p == 2:
        #     image = self.VerticalFlip(image)
        #     depth = self.VerticalFlip(depth)
        #     target = self.VerticalFlip(target)    
        # elif p == 3:
        #     image = self.rotate(image)
        #     depth = self.rotate(depth)
        #     target = self.rotate(target)  

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(Max,Min)}
        return sample

class NYU_v2_full_hr(Dataset):
    """
        NYU v2 Dataset.
        fullsize 480 x 640
        input is 480 x 640 by bicubic
        No augmentation
    """

    def __init__(self, depths, images, scale=8, transform=None):
        self.scale = scale
        self.transform = transform
        
        self.depths = depths
        self.images = images

    def __len__(self):
        return self.depths.shape[2]

    def __getitem__(self, idx):
        depth = self.depths[:,:,idx]
        image = self.images[:,:,:,idx]
        Max = depth.max()
        Min = depth.min()  
        depth = (depth-Min)/(Max-Min)

        h, w = depth.shape
        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w, h), Image.BICUBIC))
        # print(depth[:4,:8])
        # print(target[:4,:8])

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(Max,Min)}
        
        return sample

class NYU_v2_full_hr_aug(Dataset):
    """
        NYU v2 Dataset.
        fullsize 480 x 640
        input is 480 x 640
        augmentation
    """
    def __init__(self, depths, images, scale=8, transform=None):
        self.depths = depths
        self.images = images
        self.scale = scale
        self.transform = transform
        
        self.HorizontalFlip = transforms.RandomHorizontalFlip(p=1)
        self.VerticalFlip = transforms.RandomVerticalFlip(p=1)
        self.rotate = transforms.RandomRotation(180)


    def __len__(self):
        return self.depths.shape[2]

    def __getitem__(self, idx):
        depth = self.depths[:,:,idx]
        image = self.images[:,:,:,idx]
        # image = np.float32(image)
        # image /= 255
        # image = np.transpose(image,(2,0,1))
        Max = depth.max()
        Min = depth.min()  
        depth = (depth-Min)/(Max-Min)

        h, w = depth.shape
        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w, h), Image.BICUBIC))
        # image = Image.fromarray(image)
        # print(depth[:4,:8])
        # print(target[:4,:8])
        image = Image.fromarray(image)
        depth = Image.fromarray(depth)
        target = Image.fromarray(target)  

        p = random.randint(1,2)
        if p == 1:
            image = self.HorizontalFlip(image)
            depth = self.HorizontalFlip(depth)
            target = self.HorizontalFlip(target)          
        # elif p == 2:
        #     image = self.VerticalFlip(image)
        #     depth = self.VerticalFlip(depth)
        #     target = self.VerticalFlip(target)    
        # elif p == 3:
        #     image = self.rotate(image)
        #     depth = self.rotate(depth)
        #     target = self.rotate(target)  

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(Max,Min)}
        return sample

class NYU_v2_full_patch(Dataset):
    """
        NYU v2 Dataset.
        fullsize 480 x 640
        input is hr patch 192 x 192
        No augmentation
    """

    def __init__(self, depths, images, scale=8, transform=None):
        self.scale = scale
        self.transform = transform
        
        self.depths = depths
        self.images = images

    def __len__(self):
        return self.depths.shape[2]

    def __getitem__(self, idx):
        depth = self.depths[:,:,idx]
        image = self.images[:,:,:,idx]

        Max = depth.max()
        Min = depth.min()  
        depth = (depth-Min)/(Max-Min)

        s = self.scale
        h, w = depth.shape
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w, h), Image.BICUBIC))
        # print(depth[:4,:8])
        # print(target[:4,:8])

        y = random.randint(0,h-96-1)
        x = random.randint(0,w-96-1)
        # print(y,x)
        depth = depth[y:y+96, x:x+96]
        target = target[y:y+96, x:x+96]
        image = image[y:y+96, x:x+96,:]

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(Max,Min)}
        
        return sample

class Middlebury_full_lr(Dataset):
    """
        Middlebury Dataset.
        fullsize 
        input is lr 
        No augmentation
    """
    def __init__(self, scale=8, transform=None):
        self.scale = scale
        self.transform = transform

    def __len__(self):
        return 30

    def __getitem__(self, idx):
        depth_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/01_Middlebury_Dataset/Middlebury_%02d_output_depth.png"%(idx+1)
        image_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/01_Middlebury_Dataset/Middlebury_%02d_output_color.png"%(idx+1)
        depth = Image.open(depth_path)
        image = Image.open(image_path)


        depth = np.array(depth)

        h, w = depth.shape
        s = self.scale
        h = h-h%s
        w = w-w%s

  
        image = np.array(image.resize((w,h),Image.BICUBIC))
        image = self.transform(image).float()
        depth = np.array(Image.fromarray(depth).resize((w,h),Image.BICUBIC))


        # depth = depth/255
        Max = depth.max()
        Min = depth.min()  
        depth = (depth-Min)/(Max-Min)

        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))

        target = self.transform(target).float()
        depth = self.transform(depth).float()
        # print(depth[:4,:8])
        # print(target[:4,:8])

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(Max,Min)}
        
        return sample

class Middlebury_full_hr(Dataset):
    """
        Middlebury Dataset.
        fullsize 
        input is lr 
        No augmentation
    """
    def __init__(self, scale=8, transform=None):
        self.scale = scale
        self.transform = transform

    def __len__(self):
        return 30

    def __getitem__(self, idx):
        depth_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/01_Middlebury_Dataset/Middlebury_%02d_output_depth.png"%(idx+1)
        image_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/01_Middlebury_Dataset/Middlebury_%02d_output_color.png"%(idx+1)
        depth = Image.open(depth_path)
        image = Image.open(image_path)


        depth = np.array(depth)

        h, w = depth.shape
        s = self.scale
        h = h-h%s
        w = w-w%s

  
        image = np.array(image.resize((w,h),Image.BICUBIC))
        image = self.transform(image).float()
        depth = np.array(Image.fromarray(depth).resize((w,h),Image.BICUBIC))


        depth = depth/255

        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w,h),Image.BICUBIC))

        target = self.transform(target).float()
        depth = self.transform(depth).float()
        # print(depth[:4,:8])
        # print(target[:4,:8])

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(255,0)}
        
        return sample

class Middlebury_full_hr_lr(Dataset):
    """
        Middlebury Dataset.
        fullsize 
        input is lr 
        No augmentation
    """
    def __init__(self, scale=8, transform=None):
        self.scale = scale
        self.transform = transform

    def __len__(self):
        return 30

    def __getitem__(self, idx):
        depth_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/01_Middlebury_Dataset/Middlebury_%02d_output_depth.png"%(idx+1)
        image_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/01_Middlebury_Dataset/Middlebury_%02d_output_color.png"%(idx+1)
        depth = Image.open(depth_path)
        image = Image.open(image_path)


        depth = np.array(depth)

        h, w = depth.shape
        s = self.scale
        h = h-h%s
        w = w-w%s

  
        image = np.array(image.resize((w,h),Image.BICUBIC))
        image = self.transform(image).float()
        depth = np.array(Image.fromarray(depth).resize((w,h),Image.BICUBIC))

        depth = depth/255
        depth_lr = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))


        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w,h),Image.BICUBIC))
        target = self.transform(target).float()
        depth = self.transform(depth).float()
        depth_lr = self.transform(np.expand_dims(depth_lr,2)).float()
        # print(depth[:4,:8])
        # print(target[:4,:8])

        sample = {'guidance': image, 'lr':depth_lr, 'target': target, 'gt': depth, 'maxmin':(255,0)}
        
        return sample

class Lu_full_lr(Dataset):
    """
        Lu Dataset.
        fullsize 
        input is lr 
        No augmentation
    """
    def __init__(self, scale=8, transform=None):
        self.scale = scale
        self.transform = transform

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        depth_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/03_RGBD_Dataset/RGBD_%02d_output_depth.png"%(idx+1)
        image_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/03_RGBD_Dataset/RGBD_%02d_ouput_color.png"%(idx+1)
        depth = Image.open(depth_path)
        image = Image.open(image_path)


        depth = np.array(depth)



        h, w = depth.shape
        s = self.scale
        h = h-h%s
        w = w-w%s

  
        image = np.array(image.resize((w,h),Image.BICUBIC))
        image = self.transform(image).float()

        depth = np.array(Image.fromarray(depth).resize((w,h),Image.BICUBIC))
        depth = depth/255

        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))
        target = self.transform(target).float()
        depth = self.transform(depth).float()
        # print(depth[:4,:8])
        # print(target[:4,:8])

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(255,0)}
        
        return sample

class Lu_full_hr(Dataset):
    """
        Lu Dataset.
        fullsize 
        input is lr 
        No augmentation
    """
    def __init__(self, scale=8, transform=None):
        self.scale = scale
        self.transform = transform

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        depth_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/03_RGBD_Dataset/RGBD_%02d_output_depth.png"%(idx+1)
        image_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/03_RGBD_Dataset/RGBD_%02d_ouput_color.png"%(idx+1)
        depth = Image.open(depth_path)
        image = Image.open(image_path)


        depth = np.array(depth)



        h, w = depth.shape
        s = self.scale
        h = h-h%s
        w = w-w%s

  
        image = np.array(image.resize((w,h),Image.BICUBIC))
        image = self.transform(image).float()

        depth = np.array(Image.fromarray(depth).resize((w,h),Image.BICUBIC))
        depth = depth/255

        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w,h),Image.BICUBIC))
        target = self.transform(target).float()
        depth = self.transform(depth).float()
        # print(depth[:4,:8])
        # print(target[:4,:8])

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(255,0)}
        
        return sample

class Lu_full_hr_lr(Dataset):
    """
        Lu Dataset.
        fullsize 
        input is lr 
        No augmentation
    """
    def __init__(self, scale=8, transform=None):
        self.scale = scale
        self.transform = transform

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        depth_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/03_RGBD_Dataset/RGBD_%02d_output_depth.png"%(idx+1)
        image_path = "D:/2021-2/Lu and Middleburry/Depth_Enh/Depth_Enh/03_RGBD_Dataset/RGBD_%02d_ouput_color.png"%(idx+1)
        depth = Image.open(depth_path)
        image = Image.open(image_path)


        depth = np.array(depth)



        h, w = depth.shape
        s = self.scale
        h = h-h%s
        w = w-w%s

  
        image = np.array(image.resize((w,h),Image.BICUBIC))
        image = self.transform(image).float()

        depth = np.array(Image.fromarray(depth).resize((w,h),Image.BICUBIC))
        depth = depth/255
        depth_lr = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))


        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w,h),Image.BICUBIC))
        target = self.transform(target).float()
        depth = self.transform(depth).float()
        depth_lr = self.transform(np.expand_dims(depth_lr,2)).float()
        # print(depth[:4,:8])
        # print(target[:4,:8])

        sample = {'guidance': image, 'lr':depth_lr, 'target': target, 'gt': depth, 'maxmin':(255,0)}
        
        return sample


class Sintel_full_lr(Dataset):
    """
        Sintel Dataset.
        fullsize 
        input is lr 
        No augmentation
    """
    def __init__(self, scale=8, transform=None):
        self.scale = scale
        self.transform = transform
        self.frame_number = [0,50,50,21,33,50,20,50,50,50,50,50,50,50,50,50,40,50,50,50,50,50,50,50,0] 

    def __len__(self):
        return sum(self.frame_number)

    def __getitem__(self, idx):
        for i in range(24):
            s = sum(self.frame_number[:i+1])
            if idx < s:
                divide = i 
                frame =  idx-sum(self.frame_number[:i])+1
                break
        
        depth_path = "D:/2021-2/Sintel/MPI-Sintel-stereo-training-20150305/training/disparities_viz/%d/frame_%04d.png"%(divide,frame)
        image_path = "D:/2021-2/Sintel/MPI-Sintel-stereo-training-20150305/training/final_left/%d/frame_%04d.png"%(divide,frame)
        depth = Image.open(depth_path)
        image = Image.open(image_path)


        depth = np.array(depth)

        # Max = depth.max()
        # Min = depth.min()  
        # depth = (depth-Min)/(Max-Min)

        h, w = depth.shape
        s = self.scale
        h = h-h%s
        w = w-w%s

  
        image = np.array(image.resize((w,h),Image.BICUBIC))
        image = self.transform(image).float()
        depth = np.array(Image.fromarray(depth).resize((w,h),Image.BICUBIC))

        depth = depth/255

        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))
        target = self.transform(target).float()
        depth = self.transform(depth).float()
        # print(depth.max())
        # print(depth.min())
        # print(depth[:4,:8])
        # print(target[:4,:8])

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(255,0)}
        
        return sample


class Sintel_full_hr(Dataset):
    """
        Sintel Dataset.
        fullsize 
        input is lr 
        No augmentation
    """
    def __init__(self, scale=8, transform=None):
        self.scale = scale
        self.transform = transform
        self.frame_number = [0,50,50,21,33,50,20,50,50,50,50,50,50,50,50,50,40,50,50,50,50,50,50,50,0] 

    def __len__(self):
        return sum(self.frame_number)

    def __getitem__(self, idx):
        for i in range(24):
            s = sum(self.frame_number[:i+1])
            if idx < s:
                divide = i 
                frame =  idx-sum(self.frame_number[:i])+1
                break
        
        depth_path = "D:/2021-2/Sintel/MPI-Sintel-stereo-training-20150305/training/disparities_viz/%d/frame_%04d.png"%(divide,frame)
        image_path = "D:/2021-2/Sintel/MPI-Sintel-stereo-training-20150305/training/final_left/%d/frame_%04d.png"%(divide,frame)
        depth = Image.open(depth_path)
        image = Image.open(image_path)


        depth = np.array(depth)

        # Max = depth.max()
        # Min = depth.min()  
        # depth = (depth-Min)/(Max-Min)

        h, w = depth.shape
        s = self.scale
        h = h-h%s
        w = w-w%s

  
        image = np.array(image.resize((w,h),Image.BICUBIC))
        image = self.transform(image).float()
        depth = np.array(Image.fromarray(depth).resize((w,h),Image.BICUBIC))

        depth = depth/255

        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w,h),Image.BICUBIC))
        target = self.transform(target).float()
        depth = self.transform(depth).float()
        # print(depth.max())
        # print(depth.min())
        # print(depth[:4,:8])
        # print(target[:4,:8])

        sample = {'guidance': image, 'target': target, 'gt': depth, 'maxmin':(255,0)}
        
        return sample

class Sintel_full_hr_lr(Dataset):
    """
        Sintel Dataset.
        fullsize 
        input is lr 
        No augmentation
    """
    def __init__(self, scale=8, transform=None):
        self.scale = scale
        self.transform = transform
        self.frame_number = [0,50,50,21,33,50,20,50,50,50,50,50,50,50,50,50,40,50,50,50,50,50,50,50,0] 

    def __len__(self):
        return sum(self.frame_number)

    def __getitem__(self, idx):
        for i in range(24):
            s = sum(self.frame_number[:i+1])
            if idx < s:
                divide = i 
                frame =  idx-sum(self.frame_number[:i])+1
                break
        
        depth_path = "D:/2021-2/Sintel/MPI-Sintel-stereo-training-20150305/training/disparities_viz/%d/frame_%04d.png"%(divide,frame)
        image_path = "D:/2021-2/Sintel/MPI-Sintel-stereo-training-20150305/training/final_left/%d/frame_%04d.png"%(divide,frame)
        depth = Image.open(depth_path)
        image = Image.open(image_path)


        depth = np.array(depth)

        # Max = depth.max()
        # Min = depth.min()  
        # depth = (depth-Min)/(Max-Min)

        h, w = depth.shape
        s = self.scale
        h = h-h%s
        w = w-w%s

  
        image = np.array(image.resize((w,h),Image.BICUBIC))
        image = self.transform(image).float()
        depth = np.array(Image.fromarray(depth).resize((w,h),Image.BICUBIC))

        depth = depth/255
        depth_lr = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))


        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w,h),Image.BICUBIC))
        target = self.transform(target).float()
        depth = self.transform(depth).float()
        depth_lr = self.transform(np.expand_dims(depth_lr,2)).float()
        # print(depth.max())
        # print(depth.min())
        # print(depth[:4,:8])
        # print(target[:4,:8])

        sample = {'guidance': image, 'lr':depth_lr, 'target': target, 'gt': depth, 'maxmin':(255,0)}

        return sample

class NYU_v2_full_hr_lr(Dataset):
    """
        NYU v2 Dataset.
        fullsize 480 x 640
        input is 480 x 640 by bicubic
        No augmentation
    """

    def __init__(self, depths, images, scale=8, transform=None):
        self.scale = scale
        self.transform = transform
        
        self.depths = depths
        self.images = images

    def __len__(self):
        return self.depths.shape[2]

    def __getitem__(self, idx):
        depth = self.depths[:,:,idx]
        image = self.images[:,:,:,idx]
        Max = depth.max()
        Min = depth.min()  
        depth = (depth-Min)/(Max-Min)

        h, w = depth.shape
        s = self.scale
        depth_lr = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w, h), Image.BICUBIC))
        # print(depth[:4,:8])
        # print(target[:4,:8])

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth,2)).float()
            depth_lr = self.transform(np.expand_dims(depth_lr,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()

        sample = {'guidance': image, 'lr':depth_lr, 'target': target, 'gt': depth, 'maxmin':(Max,Min)}
        
        return sample