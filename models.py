import torch
import torch.nn.functional as F
import torch.nn as nn

def grid_generator(k, r, n):
    """grid_generator
    Parameters
    ---------
    r : filter_size, int
    k: kernel_size, int
    n: number of grid, int
    Returns
    -------
    torch.Tensor. shape = (n, 2, k, k)
    """
    grid_x, grid_y = torch.meshgrid([torch.linspace(k//2, k//2+r-1, steps=r),
                                     torch.linspace(k//2, k//2+r-1, steps=r)])
    # print(grid_x)
    # print(grid_y)
    grid = torch.stack([grid_x,grid_y],2).view(r,r,2)
    # print(grid)
    return grid.unsqueeze(0).repeat(n,1,1,1).cuda()


class Kernel_DKN(nn.Module):
    def __init__(self, input_channel, kernel_size):
        super(Kernel_DKN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 7)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=(2,2))
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2, stride=(2,2))
        self.conv5 = nn.Conv2d(64, 128, 5)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv7 = nn.Conv2d(128, 128, 3)
        
        self.conv_weight = nn.Conv2d(128, kernel_size**2, 1)
        self.conv_offset = nn.Conv2d(128, 2*kernel_size**2, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))
        
        return weight, offset # 1*9*120*80, 1*18*120*80
    
class DKN(nn.Module):
    def __init__(self, kernel_size, filter_size, residual=True):
        super(DKN, self).__init__()
        self.ImageKernel = Kernel_DKN(input_channel=3, kernel_size=kernel_size)
        self.DepthKernel = Kernel_DKN(input_channel=1, kernel_size=kernel_size)
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        
    def forward(self, x):
        image, depth = x
        
        weight, offset = self._shift_and_stitch(x)
        
        h, w = image.size(2), image.size(3)
        b = image.size(0)
        k = self.filter_size # 15
        r = self.kernel_size # 3
        hw = h*w # 480*320
        
        # weighted average
        # (b, 2*r**2, h, w) -> (b*hw, r, r, 2)
        offset = offset.permute(0,2,3,1).contiguous().view(b*hw, r,r, 2) # 153600*3*3*2
        # (b, r**2, h, w) -> (b*hw, r**2, 1)
        weight = weight.permute(0,2,3,1).contiguous().view(b*hw, r*r, 1) # 153600*9*1
        
        # (b*hw, r, r, 2)
        grid = grid_generator(k, r, b*hw) # 153600*3*3*2

        coord = grid + offset
        coord = (coord / k * 2) -1
        
        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_col = F.unfold(depth, k, padding=k//2).permute(0,2,1).contiguous().view(b*hw, 1, k,k)
        # 153600*1*15*15

        # (b*hw, 1, k, k), (b*hw, r, r, 2) => (b*hw, 1, r^2)
        depth_sampled = F.grid_sample(depth_col, coord).view(b*hw, 1, -1)
        # 153600*1*9

        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h,w)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h,w)

        if self.residual:
            out += depth

        return out
    
    def _infer(self, x):
        image, depth = x
        imkernel, imoffset = self.ImageKernel(image)
        depthkernel, depthoffset = self.DepthKernel(depth)
        
        weight = imkernel * depthkernel # 1*9*120*80
        offset = imoffset * depthoffset # 1*18*120*80
        
        if self.residual:
            weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight /= torch.sum(weight, 1).unsqueeze(1).expand_as(weight)            
        
        return weight, offset # 1*9*120*80, 1*18*120*80
        
    def _shift_and_stitch(self, x):
        image, depth = x
        
        offset = torch.zeros((image.size(0), 2*self.kernel_size**2, image.size(2), image.size(3)),
                             dtype=image.dtype, layout=image.layout, device=image.device)
        weight = torch.zeros((image.size(0), self.kernel_size**2, image.size(2), image.size(3)),
                             dtype=image.dtype, layout=image.layout, device=image.device)
        
        for i in range(4):
            for j in range(4):
                m = nn.ZeroPad2d((22+3-j,22+j,22+3-i,22+i))        
                img_shift = m(image)
                depth_shift = m(depth)
                w, o = self._infer( (img_shift, depth_shift) )
        
                weight[:,:,i::4,j::4] = w
                offset[:,:,i::4,j::4] = o
                
        return weight, offset # 1*9*480*320, 1*18*480*320

class MY_DKN(nn.Module):
    def __init__(self, input_channel, kernel_size):
        super(MY_DKN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 7)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=(2,2))
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2, stride=(2,2))
        self.conv5 = nn.Conv2d(64, 128, 5)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv7 = nn.Conv2d(128, 128, 3)
        
        self.conv_weight = nn.Conv2d(128, kernel_size**2, 1)
        # self.conv_offset = nn.Conv2d(128, 2*kernel_size**2, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))
        
        return weight # 1*9*120*80
    
class MINE(nn.Module):
    def __init__(self, kernel_size, filter_size, residual=True):
        super(MINE, self).__init__()
        self.ImageKernel = MY_DKN(input_channel=3, kernel_size=kernel_size)
        self.DepthKernel = MY_DKN(input_channel=1, kernel_size=kernel_size)
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        
    def forward(self, x):
        image, depth = x
        
        h, w = image.size(2), image.size(3)
        b = image.size(0)
        k = self.filter_size # 15
        r = self.kernel_size # 3
        hw = h*w # 480*320
        
        # weighted average
        weight_project = self._shift_and_stitch(x)
        weight_accept = torch.zeros(weight_project.shape, device=weight_project.device)  

        weight_accept[:,8,:-1,:-1] = weight_project[:,0,1:,1:]
        weight_accept[:,7,:-1,:]  = weight_project[:,1,1:,:]
        weight_accept[:,6,:-1,1:]  = weight_project[:,2,1:,:-1]
        weight_accept[:,5,:,:-1]   = weight_project[:,3,:,1:]        
        weight_accept[:,4,:,:]     = weight_project[:,4,:,:]
        weight_accept[:,3,:,1:]    = weight_project[:,5,:,:-1]
        weight_accept[:,2,1:,:-1]  = weight_project[:,6,:-1,1:]
        weight_accept[:,1,1:,:]    = weight_project[:,7,:-1,:]
        weight_accept[:,0,1:,1:]   = weight_project[:,8,:-1,:-1]

        # (b, r**2, h, w) -> (b*hw, r**2, 1)
        weight_accept = weight_accept.permute(0,2,3,1).contiguous() # .view(b*hw, r*r, 1) # 1*480*320*9
        weight_accept /= (weight_accept.sum(dim = 3, keepdim= True) + 0.0000000001)


        
        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_unfold = F.unfold(depth, r, padding=r//2).permute(0,2,1).contiguous().view(b,h,w,r**2)
        # 153600*1*3,3

        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h,w)
        # out = torch.bmm(depth_unfold, weight).view(b, 1, h,w)
        out = (depth_unfold * weight_accept).sum(dim = 3).unsqueeze(dim = 1)

        if self.residual:
            out += depth

        return out
    
    def _infer(self, x):
        image, depth = x
        imkernel = self.ImageKernel(image)
        depthkernel = self.DepthKernel(depth)
        
        weight = imkernel * depthkernel # 1*9*120*80
        
        if self.residual:
            weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight /= torch.sum(weight, 1).unsqueeze(1).expand_as(weight)            
        
        return weight # 1*9*120*80
        
    def _shift_and_stitch(self, x):
        image, depth = x
        
        weight = torch.zeros((image.size(0), self.kernel_size**2, image.size(2), image.size(3)),
                             dtype=image.dtype, layout=image.layout, device=image.device)
        
        for i in range(4):
            for j in range(4):
                m = nn.ZeroPad2d((22+3-j,22+j,22+3-i,22+i))        
                img_shift = m(image)
                depth_shift = m(depth)
                w = self._infer( (img_shift, depth_shift) )
        
                weight[:,:,i::4,j::4] = w
                
        return weight # 1*9*480*320

class MY_DKN_2(nn.Module):
    def __init__(self, input_channel, kernel_size):
        super(MY_DKN_2, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 7)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=(2,2))
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2, stride=(2,2))
        self.conv5 = nn.Conv2d(64, 128, 5)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv7 = nn.Conv2d(128, 128, 3)
        
        self.conv_weight = nn.Conv2d(128, kernel_size**2, 1)
        # self.conv_offset = nn.Conv2d(128, 2*kernel_size**2, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))
        
        return weight # 1*9*120*80
    
class MINE_2(nn.Module):
    def __init__(self, kernel_size, filter_size, residual=True):
        super(MINE_2, self).__init__()
        self.ImageKernel = MY_DKN_2(input_channel=4, kernel_size=kernel_size)
        # self.DepthKernel = MY_DKN(input_channel=1, kernel_size=kernel_size)
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        
    def forward(self, x):
        image, depth = x
        
        h, w = image.size(2), image.size(3)
        b = image.size(0)
        k = self.filter_size # 15
        r = self.kernel_size # 3
        hw = h*w # 480*320
        
        # weighted average
        weight_project = self._shift_and_stitch(x)
        # eight_project.backward()
        weight_accept = torch.zeros(weight_project.shape, device=weight_project.device)  

        weight_accept[:,8,:-1,:-1] = weight_project[:,0,1:,1:]
        weight_accept[:,7,:-1,:]  = weight_project[:,1,1:,:]
        weight_accept[:,6,:-1,1:]  = weight_project[:,2,1:,:-1]
        weight_accept[:,5,:,:-1]   = weight_project[:,3,:,1:]        
        weight_accept[:,4,:,:]     = weight_project[:,4,:,:]
        weight_accept[:,3,:,1:]    = weight_project[:,5,:,:-1]
        weight_accept[:,2,1:,:-1]  = weight_project[:,6,:-1,1:]
        weight_accept[:,1,1:,:]    = weight_project[:,7,:-1,:]
        weight_accept[:,0,1:,1:]   = weight_project[:,8,:-1,:-1]

        # (b, r**2, h, w) -> (b*hw, r**2, 1)
        weight_accept = weight_accept.permute(0,2,3,1).contiguous() # .view(b*hw, r*r, 1) # 1*480*320*9
        W = weight_accept.sum(dim = 3, keepdim= True) + 0.0000000001
        weight_accept = weight_accept/W


        
        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_unfold = F.unfold(depth, r, padding=r//2).permute(0,2,1).contiguous().view(b,h,w,r**2)
        # 153600*1*3,3

        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h,w)
        # out = torch.bmm(depth_unfold, weight).view(b, 1, h,w)
        out = (depth_unfold * weight_accept).sum(dim = 3).unsqueeze(dim = 1)

        if self.residual:
            out = out + depth

        return out
    
    def _infer(self, x):
        image, depth = x
        input_4 = torch.cat([image,depth],1)
        imkernel = self.ImageKernel(input_4)
        # depthkernel = self.DepthKernel(depth)
        
        weight = imkernel  # 1*9*120*80
        
        if self.residual:
            weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight /= torch.sum(weight, 1).unsqueeze(1).expand_as(weight)            
        
        return imkernel # 1*9*120*80
        
    def _shift_and_stitch(self, x):
        image, depth = x
        
        weight = torch.zeros((image.size(0), self.kernel_size**2, image.size(2), image.size(3)),
                             dtype=image.dtype, layout=image.layout, device=image.device)
        
        for i in range(4):
            for j in range(4):
                m = nn.ZeroPad2d((22+3-j,22+j,22+3-i,22+i))        
                img_shift = m(image)
                depth_shift = m(depth)
                w = self._infer( (img_shift, depth_shift) )
        
                weight[:,:,i::4,j::4] = w
                
        return weight # 1*9*480*320



def resample_data(input, s, bgr=True, rgb=False):
    """
        input: torch.floatTensor (N, C, H, W)
        s: int (resample factor)
    """    
    
    assert( not input.size(2)%s and not input.size(3)%s)
    
    if input.size(1) == 3:
        if bgr == True:
        # bgr2gray (same as opencv conversion matrix)
            input = (0.299 * input[:,2] + 0.587 * input[:,1] + 0.114 * input[:,0]).unsqueeze(1)
        if rgb == True:
        # bgr2gray (same as opencv conversion matrix)
            input = (0.299 * input[:,0] + 0.587 * input[:,1] + 0.114 * input[:,2]).unsqueeze(1)
        
    out = torch.cat([input[:,:,i::s,j::s] for i in range(s) for j in range(s)], dim=1)

    """
        out: torch.floatTensor (N, s**2, H/s, W/s)
    """
    return out


class Kernel_FDKN(nn.Module):
    def __init__(self, input_channel, kernel_size, factor=4):
        super(Kernel_FDKN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.conv_weight = nn.Conv2d(128, kernel_size**2*(factor)**2, 1)
        self.conv_offset = nn.Conv2d(128, 2*kernel_size**2*(factor)**2, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))

        offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))
        
        return weight, offset


class FDKN(nn.Module):
    def __init__(self, kernel_size, filter_size, residual=True):
        super(FDKN, self).__init__()
        self.factor = 4 # resample factor
        self.ImageKernel = Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.DepthKernel = Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        
    def forward(self, x):
        image, depth = x
        
        re_im = resample_data(image, self.factor, bgr=False, rgb=True)
        re_dp = resample_data(depth, self.factor)
        
        imkernel, imoffset       = self.ImageKernel(re_im)
        depthkernel, depthoffset = self.DepthKernel(re_dp)
        
        weight = imkernel * depthkernel
        offset = imoffset * depthoffset
        
        ps = nn.PixelShuffle(4)
        weight = ps(weight)
        offset = ps(offset)
        
        if self.residual:
            weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight /= torch.sum(weight, 1).unsqueeze(1).expand_as(weight)            
            
        b, h, w = image.size(0), image.size(2), image.size(3)
        k = self.filter_size
        r = self.kernel_size
        hw = h*w
        
        # weighted average
        # (b, 2*r**2, h, w) -> (b*hw, r, r, 2)
        offset = offset.permute(0,2,3,1).contiguous().view(b*hw, r,r, 2)
        # (b, r**2, h, w) -> (b*hw, r**2, 1)
        weight = weight.permute(0,2,3,1).contiguous().view(b*hw, r*r, 1)
        
        # (b*hw, r, r, 2)
        grid = grid_generator(k, r, b*hw)
        coord = grid + offset
        coord = (coord / k * 2) -1
        
        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_col = F.unfold(depth, k, padding=k//2).permute(0,2,1).contiguous().view(b*hw, 1, k,k)
        
        # (b*hw, 1, k, k), (b*hw, r, r, 2) => (b*hw, 1, r^2)
        depth_sampled = F.grid_sample(depth_col, coord).view(b*hw, 1, -1)
        
        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h, w)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h,w)

        if self.residual:
            out += depth
        
        return out
    
def my_grid_generator(k, h, w):

    grid = torch.zeros(k,k)  
    for i in range(k):
        for j in range (k):
            grid[i,j] = 1-(((i+0.5-k/2)**2+(j+0.5-k/2)**2)**0.5)/k
    
    grid = grid.repeat(h//k,w//k)
    # print(grid)
    return grid.unsqueeze(0).unsqueeze(0).cuda()



class MY_Kernel_FDKN(nn.Module):
    def __init__(self, input_channel, kernel_size, factor=4):
        super(MY_Kernel_FDKN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv_weight = nn.Conv2d(256, kernel_size**2*(factor)**2, 1)
        # self.conv_offset = nn.Conv2d(128, 2*kernel_size**2*(factor)**2, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.relu(self.conv8(x))

        # offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))*2-1
        
        return weight

class MY_FDKN(nn.Module):
    def __init__(self, kernel_size, filter_size, residual=True):
        super(MY_FDKN, self).__init__()
        self.factor = 4 # resample factor
        self.ImageKernel = MY_Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.DepthKernel = MY_Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.ImageKernel1 = MY_Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.DepthKernel1 = MY_Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.ImageKernel2 = MY_Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.DepthKernel2 = MY_Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)    
        
    def forward(self, x):
        image, depth = x
        b, h, w = image.size(0), image.size(2), image.size(3)
        k = self.filter_size
        r = self.kernel_size
        hw = h*w

        re_im = resample_data(image, self.factor)
        re_dp = resample_data(depth, self.factor)
        
        imkernel    = self.ImageKernel(re_im)
        depthkernel = self.DepthKernel(re_dp)
        
        weight = imkernel * depthkernel
      
        
        ps = nn.PixelShuffle(4)
        weight = ps(weight)
     
        
        # if self.residual:
        #     weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        # else:
        #     weight /= torch.sum(weight, 1).unsqueeze(1).expand_as(weight)            

        weight = weight.permute(0,2,3,1).contiguous() # .view(b*hw, r*r, 1) # 1*480*320*9
        # # weight_accept /= (weight_accept.sum(dim = 3, keepdim= True) + 0.0000000001)


        
        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_unfold = F.unfold(depth, r, padding=r//2).permute(0,2,1).contiguous().view(b,h,w,r**2)
        # 153600*1*3,3

        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h,w)
        # out = torch.bmm(depth_unfold, weight).view(b, 1, h,w)
        out = (depth_unfold * weight).sum(dim = 3).unsqueeze(dim = 1)

        if self.residual:
            out += depth

        re_dp = resample_data(out, self.factor)
        
        imkernel    = self.ImageKernel1(re_im)
        depthkernel = self.DepthKernel1(re_dp)
        
        weight = imkernel * depthkernel
  
        weight = ps(weight)
        weight= weight.permute(0,2,3,1).contiguous()
        # depth_unfold2 = F.unfold(depth, r, padding=r//2).permute(0,2,1).contiguous().view(b,h,w,r**2)
        out = (depth_unfold * weight).sum(dim = 3).unsqueeze(dim = 1)

        if self.residual:
            out += depth        

        re_dp = resample_data(out, self.factor)
        
        imkernel    = self.ImageKernel2(re_im)
        depthkernel = self.DepthKernel2(re_dp)
        
        weight = imkernel * depthkernel
  
        weight = ps(weight)
        weight= weight.permute(0,2,3,1).contiguous()
        # depth_unfold2 = F.unfold(depth, r, padding=r//2).permute(0,2,1).contiguous().view(b,h,w,r**2)
        out = (depth_unfold * weight).sum(dim = 3).unsqueeze(dim = 1)

        if self.residual:
            out += depth      
                  
        return out

class Generator(nn.Module):
    def __init__(self, input_channel, kernel_size, factor=4):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, 16, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.ps1 = nn.PixelShuffle(2)

        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.ps2 = nn.PixelShuffle(2)

        self.conv7 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv8_bn = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9_bn = nn.BatchNorm2d(256)
        self.ps3      = nn.PixelShuffle(2)

        self.conv10 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv10_bn = nn.BatchNorm2d(32)
        self.conv11 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv11_bn = nn.BatchNorm2d(16)
        self.conv12 = nn.Conv2d(16, 1, 3, padding=1)
      
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.ps1(x)

        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = self.ps2(x)

        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.relu(self.conv9_bn(self.conv9(x)))
        x = self.ps3(x)

        x = F.relu(self.conv10_bn(self.conv10(x)))
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12(x))
        
        return x

    
class Discriminator(nn.Module):
    def __init__(self, input_channel, batch_size):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.main = nn.Sequential(
            # 输入大小 (1) x 480 x 320
            nn.Conv2d(input_channel, input_channel*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (4) x 240 x 160
            nn.Conv2d(input_channel*4, input_channel*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel*16),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (16) x 120 x 80
            nn.Conv2d(input_channel*16, input_channel*64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel*64),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (64) x 60 x 40
            nn.Conv2d(input_channel*64, input_channel*256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel*256),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (256) x 30 x 20
            nn.Conv2d(input_channel*256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (256) x 15 x 10
            nn.Conv2d(256, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (64) x 15 x 10
            nn.Conv2d(64, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (16) x 15 x 10
            nn.Conv2d(16, 1, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (1) x 15 x 10
        )
        self.linear = nn.Sequential(
            nn.Linear(150,50),
            nn.ReLU(),
            nn.Linear(50,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid(),            
        )

    def forward(self, input):
        x = self.main(input)
        x = x.view(self.batch_size,150)
        x = self.linear(x)
        return x

class RRDB(nn.Module):
    def __init__(self, input_channel=64, kernel_size=3, beta=0.1):
        super(RRDB, self).__init__()
        self.beta = beta
        self.conv1 = nn.Conv2d(input_channel*1, input_channel, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(input_channel*2, input_channel, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(input_channel*3, input_channel, kernel_size, padding=kernel_size//2)
        self.conv4 = nn.Conv2d(input_channel*4, input_channel, kernel_size, padding=kernel_size//2)
        self.conv5 = nn.Conv2d(input_channel*5, input_channel, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat((x,x1),dim=1)))
        x3 = F.relu(self.conv3(torch.cat((x,x1,x2),dim=1)))
        x4 = F.relu(self.conv4(torch.cat((x,x1,x2,x3),dim=1)))
        x5 = self.conv5(torch.cat((x,x1,x2,x3,x4),dim=1))


        out = x + self.beta * x5
        
        return out

class MKPN(nn.Module):
    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(MKPN, self).__init__()
        self.conv = nn.Conv2d(1, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.KPN = nn.Sequential(
            nn.Conv2d(RRDB_channel, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            # nn.Conv2d((up_scale*kpn_kernel)**2, (up_scale*kpn_kernel)**2, conv_kernel_size, padding=conv_kernel_size//2),
            # # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            # nn.ReLU(),
            nn.Conv2d(RRDB_channel, (up_scale*kpn_kernel)**2, conv_kernel_size, padding=conv_kernel_size//2),
            )
        self.up_scale = up_scale
        self.ps = nn.PixelShuffle(up_scale)
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        
        
        
    def forward(self, x):
        image, depth = x
        
        x0 = self.conv(depth)
        weight = self.block1(x0)
        weight = self.block2(weight)
        weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.KPN(weight)
        weight = self.ps(weight) # B * K^2 * H * W 
        # weight = weight.permute(0,2,3,1) # B * H * W * K^2

        if self.residual:
            # weight = torch.sigmoid(weight)*2-1  # B * K^2 * H * W 
            weight = F.tanh(weight)
        else:
            weight = F.softmax(weight,dim=1) # B * K^2 * H * W 

            # weight = torch.exp(weight)
            # weight = weight/(torch.sum(weight, dim=1, keepdim=True)+1e-6)
            # print(torch.sum(weight[0,:,0,0]))
                  
            
        b, h, w = depth.size(0), depth.size(2), depth.size(3)

        # if self.residual:
        #     out += depth
    
        depth_unfold = F.unfold(depth, self.kpn_kernel, padding=self.kpn_kernel//2).view(b,self.kpn_kernel**2,h,w)
        # B * K^2 * H * W
        # print(depth_unfold[0,:,1,1])
        depth_unfold = F.interpolate(input=depth_unfold, scale_factor=self.up_scale, mode='nearest', align_corners=None)
        # print(depth_unfold[0,:,8,8])
        # print(depth_unfold[0,:,15,15])
        # print(depth[0,0,:3,:3])
        

        out = (depth_unfold * weight).sum(dim = 1).unsqueeze(dim = 1)

        if self.residual:
            out += F.interpolate(input=depth, scale_factor=self.up_scale, mode='nearest', align_corners=None)

        return out



class MKPNG(nn.Module):
    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(MKPNG, self).__init__()
        self.conv = nn.Conv2d(1, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.KPN = nn.Sequential(
            nn.Conv2d(RRDB_channel, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            # nn.Conv2d((up_scale*kpn_kernel)**2, (up_scale*kpn_kernel)**2, conv_kernel_size, padding=conv_kernel_size//2),
            # # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            # nn.ReLU(),
            nn.Conv2d(RRDB_channel, (up_scale*kpn_kernel)**2, conv_kernel_size, padding=conv_kernel_size//2),
            )

        self.conv_G = nn.Conv2d(up_scale**2, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1_G = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2_G = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block3_G = RRDB(RRDB_channel, conv_kernel_size, beta)
        
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.KPN_G = nn.Sequential(
            nn.Conv2d(RRDB_channel, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            # nn.Conv2d((up_scale*kpn_kernel)**2, (up_scale*kpn_kernel)**2, conv_kernel_size, padding=conv_kernel_size//2),
            # # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            # nn.ReLU(),
            nn.Conv2d(RRDB_channel, (up_scale*kpn_kernel)**2, conv_kernel_size, padding=conv_kernel_size//2),
            )

        self.up_scale = up_scale
        self.ps = nn.PixelShuffle(up_scale)
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        
    def forward(self, x):
        image, depth = x
        image = resample_data(image,self.up_scale,bgr=True, rgb=False)
        
        x0 = self.conv(depth)
        weight = self.block1(x0)
        weight = self.block2(weight)
        weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.KPN(weight)
        weight = self.ps(weight) # B * K^2 * H * W 

        x0_G = self.conv_G(image)
        weight_G = self.block1_G(x0_G)
        weight_G = self.block2_G(weight_G)
        weight_G = self.block3_G(weight_G)
        # weight = self.block4(weight)
        weight_G = x0_G + self.beta * weight_G
    
        weight_G = self.KPN_G(weight_G)
        weight_G = self.ps(weight_G) # B * K^2 * H * W 

        weight = weight * weight_G

        if self.residual:
            # weight = torch.sigmoid(weight)*2-1  # B * K^2 * H * W 
            # weight = F.tanh(weight)
            weight -= torch.mean(weight, dim=1, keepdim=True)
        else:
            weight = F.softmax(weight,dim=1) # B * K^2 * H * W 

            # weight = torch.exp(weight)
            # weight = weight/(torch.sum(weight, dim=1, keepdim=True)+1e-6)
            # print(torch.sum(weight[0,:,0,0]))
                  
            
        b, h, w = depth.size(0), depth.size(2), depth.size(3)

        # if self.residual:
        #     out += depth
    
        depth_unfold = F.unfold(depth, self.kpn_kernel, padding=self.kpn_kernel//2).view(b,self.kpn_kernel**2,h,w)
        # B * K^2 * H * W
        # print(depth_unfold[0,:,1,1])
        depth_unfold = F.interpolate(input=depth_unfold, scale_factor=self.up_scale, mode='nearest', align_corners=None)
        # print(depth_unfold[0,:,8,8])
        # print(depth_unfold[0,:,15,15])
        # print(depth[0,0,:3,:3])
        

        out = (depth_unfold * weight).sum(dim = 1).unsqueeze(dim = 1)

        if self.residual:
            out += F.interpolate(input=depth, scale_factor=self.up_scale, mode='nearest', align_corners=None)

        return out



class FKPNUL(nn.Module):
    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(FKPNUL, self).__init__()
        self.conv = nn.Conv2d(1, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB(RRDB_channel, conv_kernel_size, beta)
        # self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.KPN = nn.Sequential(
            nn.Conv2d(RRDB_channel, (up_scale*kpn_kernel)*2, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d((up_scale*kpn_kernel)*2, (up_scale*kpn_kernel)*2, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d((up_scale*kpn_kernel)*2, (up_scale*kpn_kernel)*2, conv_kernel_size, padding=conv_kernel_size//2),
            )
        self.up_scale = up_scale
        self.ps = nn.PixelShuffle(up_scale)
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        
        
        
    def forward(self, x):
        image, depth = x
        b, h, w = depth.size(0), depth.size(2), depth.size(3)

        x0 = self.conv(depth)
        weight = self.block1(x0)
        weight = self.block2(weight)
        # weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.KPN(weight) # B * (S*K)*2 * H * W 

        Weight = torch.zeros(b, (self.kpn_kernel*self.up_scale)**2, h, w).cuda()
        for i in range(b):
            weight_1 = weight[i,:self.kpn_kernel*self.up_scale,:,:].permute(1,2,0)
            weight_2 = weight[i,self.kpn_kernel*self.up_scale:,:,:].permute(1,2,0)
            weight_1 = weight_1.view(h*w,self.kpn_kernel*self.up_scale,1)
            weight_2 = weight_2.view(h*w,1,self.kpn_kernel*self.up_scale)
            weight_12 = torch.bmm(weight_1,weight_2).view(h,w,(self.kpn_kernel*self.up_scale)**2)
            Weight[i,:,:,:] = weight_12.permute(2,0,1)





        Weight = self.ps(Weight) # B * K^2 * H * W 
        # weight = weight.permute(0,2,3,1) # B * H * W * K^2

        if self.residual:
            Weight = torch.sigmoid(Weight)*2-1  # B * K^2 * H * W 
        else:
            Weight = F.softmax(Weight,dim=1) # B * K^2 * H * W 

            # weight = torch.exp(weight)
            # weight = weight/(torch.sum(weight, dim=1, keepdim=True)+1e-6)
            # print(torch.sum(weight[0,:,0,0]))
                  
            
        

        # if self.residual:
        #     out += depth
    
        depth_unfold = F.unfold(depth, self.kpn_kernel, padding=self.kpn_kernel//2).view(b,self.kpn_kernel**2,h,w)
        # B * K^2 * H * W
        # print(depth_unfold[0,:,1,1])
        depth_unfold = F.interpolate(input=depth_unfold, scale_factor=self.up_scale, mode='nearest', align_corners=None)
        # print(depth_unfold[0,:,8,8])
        # print(depth_unfold[0,:,15,15])
        # print(depth[0,0,:3,:3])
        

        out = (depth_unfold * Weight).sum(dim = 1).unsqueeze(dim = 1)

        if self.residual:
            out += F.interpolate(input=depth, scale_factor=self.up_scale, mode='nearest', align_corners=None)

        return out



class RRDB_baseline(nn.Module):
    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(RRDB_baseline, self).__init__()
        self.conv = nn.Conv2d(1, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.Net = nn.Sequential(
            nn.Conv2d(RRDB_channel, 8, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            # nn.Conv2d(16, 4, conv_kernel_size, padding=conv_kernel_size//2),
            # # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            # nn.ReLU(),
            nn.Conv2d(8, 1, conv_kernel_size, padding=conv_kernel_size//2),
            )
        self.up_scale = up_scale
        self.ps = nn.PixelShuffle(up_scale)
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        
        
        
    def forward(self, x):
        image, depth = x
        
        x0 = self.conv(depth)
        weight = self.block1(x0)
        weight = self.block2(weight)
        weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.Net(weight)
        # weight = self.ps(weight) # B * K^2 * H * W 
 

        return weight



class FRRDB_baseline(nn.Module):
    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(FRRDB_baseline, self).__init__()
        self.conv = nn.Conv2d(up_scale**2, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.Net = nn.Sequential(
            nn.Conv2d(RRDB_channel, up_scale**2, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(up_scale**2, up_scale**2, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(up_scale**2, up_scale**2, conv_kernel_size, padding=conv_kernel_size//2),
            )
        self.up_scale = up_scale
        self.factor = up_scale
        self.ps = nn.PixelShuffle(up_scale)
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        
        
        
    def forward(self, x):
        image, depth = x
        # re_im = resample_data(image, self.factor)
        re_dp = resample_data(depth, self.factor)

        x0 = self.conv(re_dp)
        weight = self.block1(x0)
        weight = self.block2(weight)
        weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.ps(self.Net(weight))
        # weight = self.ps(weight) # B * K^2 * H * W 
 

        return weight

class FRRDN(nn.Module): # FRRDN
    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(FRRDN, self).__init__()
        self.conv = nn.Conv2d(1, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.Net = nn.Sequential(
            nn.Conv2d(RRDB_channel, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            # nn.Conv2d(RRDB_channel, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2),
            # # nn.BatchNorm2d((up_scale*kpn_kernl)**2),
            # nn.ReLU(),
            nn.Conv2d(RRDB_channel, up_scale**2, conv_kernel_size, padding=conv_kernel_size//2),
            )
        self.up_scale = up_scale
        self.factor = up_scale
        self.ps = nn.PixelShuffle(up_scale)
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        
        
        
    def forward(self, x):
        image, depth = x
        # re_im = resample_data(image, self.factor)
        # re_dp = resample_data(depth, self.factor)

        x0 = self.conv(depth)
        weight = self.block1(x0)
        weight = self.block2(weight)
        weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.ps(self.Net(weight))
        # weight = self.ps(weight) # B * K^2 * H * W 
 

        return weight


class KPN_baseline(nn.Module):

    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(KPN_baseline, self).__init__()
        self.conv = nn.Conv2d(1, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.KPN = nn.Sequential(
            nn.Conv2d(RRDB_channel, kpn_kernel**2, conv_kernel_size, padding=conv_kernel_size//2),
            nn.ReLU(),
            # nn.Conv2d(RRDB_channel, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.ReLU(),
            nn.Conv2d(kpn_kernel**2, kpn_kernel**2, conv_kernel_size, padding=conv_kernel_size//2),
            )
        self.up_scale = up_scale
        self.ps = nn.PixelShuffle(up_scale)
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        
        
        
    def forward(self, x):
        image, depth = x
        
        x0 = self.conv(depth)
        weight = self.block1(x0)
        weight = self.block2(weight)
        weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.KPN(weight)
        # B * K^2 * H * W 


        if self.residual:
            # weight = torch.sigmoid(weight)*2-1
            weight = F.tanh(weight)
        else:
            weight = F.softmax(weight,dim=1)
            # print(torch.sum(weight[0,:,0,0]))
                  
            
        b, h, w = depth.size(0), depth.size(2), depth.size(3)

        # if self.residual:
        #     out += depth
    
        depth_unfold = F.unfold(depth, self.kpn_kernel, padding=self.kpn_kernel//2).view(b,self.kpn_kernel**2,h,w)
        # B * K^2 * H * W
        # print(depth_unfold[0,:,1,1])
        # depth_unfold = F.interpolate(input=depth_unfold, scale_factor=self.up_scale, mode='nearest', align_corners=None)
        # print(depth_unfold[0,:,8,8])
        # print(depth_unfold[0,:,15,15])
        # print(depth[0,0,:3,:3])
        

        out = (depth_unfold * weight).sum(dim = 1).unsqueeze(dim = 1)

        if self.residual:
            out += depth

        return out


class FKPN_baseline(nn.Module):

    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(FKPN_baseline, self).__init__()
        self.conv = nn.Conv2d(64, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB(RRDB_channel, conv_kernel_size, beta)
        # self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.KPN = nn.Sequential(
            nn.Conv2d(RRDB_channel, kpn_kernel**2*64, conv_kernel_size, padding=conv_kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(kpn_kernel**2*64, kpn_kernel**2*64, conv_kernel_size, padding=conv_kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(kpn_kernel**2*64, kpn_kernel**2*64, conv_kernel_size, padding=conv_kernel_size//2),
            )
        self.up_scale = up_scale
        
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        self.factor = 8 # resample_data pixel shuffle
        self.ps = nn.PixelShuffle(self.factor)
        
    def forward(self, x):
        image, depth = x
        # re_im = resample_data(image, self.factor)
        re_dp = resample_data(depth, self.factor)
        
        x0 = self.conv(re_dp)
        weight = self.block1(x0)
        weight = self.block2(weight)
        # weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.ps(self.KPN(weight))
       


        if self.residual:
            weight = torch.sigmoid(weight)*2-1
        else:
            weight = F.softmax(weight,dim=1)
            # print(torch.sum(weight[0,:,0,0]))
                  
            
        b, h, w = depth.size(0), depth.size(2), depth.size(3)

        # if self.residual:
        #     out += depth
    
        depth_unfold = F.unfold(depth, self.kpn_kernel, padding=self.kpn_kernel//2).view(b,self.kpn_kernel**2,h,w)
        # B * K^2 * H * W
        # print(depth_unfold[0,:,1,1])
        # depth_unfold = F.interpolate(input=depth_unfold, scale_factor=self.up_scale, mode='nearest', align_corners=None)
        # print(depth_unfold[0,:,8,8])
        # print(depth_unfold[0,:,15,15])
        # print(depth[0,0,:3,:3])
        

        out = (depth_unfold * weight).sum(dim = 1).unsqueeze(dim = 1)

        if self.residual:
            out += depth

        return out

class RRDB_norm(nn.Module):
    def __init__(self, input_channel=64, kernel_size=3, beta=0.1):
        super(RRDB_norm, self).__init__()
        self.beta = beta
        self.conv1 = nn.Conv2d(input_channel*1, input_channel, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(input_channel*2, input_channel, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(input_channel*3, input_channel, kernel_size, padding=kernel_size//2)
        self.conv4 = nn.Conv2d(input_channel*4, input_channel, kernel_size, padding=kernel_size//2)
        self.conv5 = nn.Conv2d(input_channel*5, input_channel, kernel_size, padding=kernel_size//2, bias=False)
        self.norm  = nn.BatchNorm2d(input_channel)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat((x,x1),dim=1)))
        x3 = F.relu(self.conv3(torch.cat((x,x1,x2),dim=1)))
        x4 = F.relu(self.conv4(torch.cat((x,x1,x2,x3),dim=1)))
        x5 = self.norm(self.conv5(torch.cat((x,x1,x2,x3,x4),dim=1)))


        out = x + self.beta * x5
        
        return out

class KPN_baseline_norm(nn.Module):
    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(KPN_baseline_norm, self).__init__()
        self.conv = nn.Conv2d(1, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.norm = nn.BatchNorm2d(RRDB_channel)
        self.block1 = RRDB_norm(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB_norm(RRDB_channel, conv_kernel_size, beta)
        # self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.KPN = nn.Sequential(
            nn.Conv2d(RRDB_channel, kpn_kernel**2, conv_kernel_size, padding=conv_kernel_size//2),
            nn.BatchNorm2d(kpn_kernel**2),
            nn.ReLU(),
            nn.Conv2d(kpn_kernel**2, kpn_kernel**2, conv_kernel_size, padding=conv_kernel_size//2),
            nn.BatchNorm2d(kpn_kernel**2),
            nn.ReLU(),
            nn.Conv2d(kpn_kernel**2, kpn_kernel**2, conv_kernel_size, padding=conv_kernel_size//2),
            )
        self.up_scale = up_scale
        self.ps = nn.PixelShuffle(up_scale)
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        
        
        
    def forward(self, x):
        image, depth = x
        
        x0 = F.relu(self.norm(self.conv(depth)))
        weight = self.block1(x0)
        weight = self.block2(weight)
        # weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.KPN(weight)
        # B * K^2 * H * W 


        if self.residual:
            weight = torch.sigmoid(weight)*2-1
        else:
            weight = F.softmax(weight,dim=1)
            # print(torch.sum(weight[0,:,0,0]))
                  
            
        b, h, w = depth.size(0), depth.size(2), depth.size(3)

        # if self.residual:
        #     out += depth
    
        depth_unfold = F.unfold(depth, self.kpn_kernel, padding=self.kpn_kernel//2).view(b,self.kpn_kernel**2,h,w)
        # B * K^2 * H * W
        # print(depth_unfold[0,:,1,1])
        # depth_unfold = F.interpolate(input=depth_unfold, scale_factor=self.up_scale, mode='nearest', align_corners=None)
        # print(depth_unfold[0,:,8,8])
        # print(depth_unfold[0,:,15,15])
        # print(depth[0,0,:3,:3])
        

        out = (depth_unfold * weight).sum(dim = 1).unsqueeze(dim = 1)

        if self.residual:
            out += depth

        return out


class RRDB_Guide(nn.Module):

    def __init__(self, up_scale=8, RRDB_channel=64, conv_kernel_size=3, beta=0.1, kpn_kernel=3, residual=True):
        super(RRDB_Guide, self).__init__()
        self.conv = nn.Conv2d(1 + up_scale**2, RRDB_channel, conv_kernel_size, padding=conv_kernel_size//2)
        self.block1 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.block2 = RRDB(RRDB_channel, conv_kernel_size, beta)
        # self.block3 = RRDB(RRDB_channel, conv_kernel_size, beta)
        # self.block4 = RRDB(RRDB_channel, conv_kernel_size, beta)
        self.KPN = nn.Sequential(
            nn.Conv2d(RRDB_channel, (up_scale*kpn_kernel)**2, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d((up_scale*kpn_kernel)**2, (up_scale*kpn_kernel)**2, conv_kernel_size, padding=conv_kernel_size//2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d((up_scale*kpn_kernel)**2, (up_scale*kpn_kernel)**2, conv_kernel_size, padding=conv_kernel_size//2),
            )
        self.up_scale = up_scale
        self.ps = nn.PixelShuffle(up_scale)
        self.beta = beta
        self.kpn_kernel = kpn_kernel
        self.residual = residual
        
        
        
    def forward(self, x):
        image, depth = x
        # print(image.max())
        image = resample_data(image,self.up_scale)

        x0 = self.conv(torch.cat((depth,image),dim=1))
        weight = self.block1(x0)
        weight = self.block2(weight)
        # weight = self.block3(weight)
        # weight = self.block4(weight)
        weight = x0 + self.beta * weight
    
        weight = self.KPN(weight)
        weight = self.ps(weight) # B * K^2 * H * W 
        # weight = weight.permute(0,2,3,1) # B * H * W * K^2

        if self.residual:
            weight = torch.sigmoid(weight)*2-1  # B * K^2 * H * W 
        else:
            weight = F.softmax(weight,dim=1) # B * K^2 * H * W 
            # print(torch.sum(weight[0,:,0,0]))
                  
            
        b, h, w = depth.size(0), depth.size(2), depth.size(3)

        # if self.residual:
        #     out += depth
    
        depth_unfold = F.unfold(depth, self.kpn_kernel, padding=self.kpn_kernel//2).view(b,self.kpn_kernel**2,h,w)
        # B * K^2 * H * W
        # print(depth_unfold[0,:,1,1])
        depth_unfold = F.interpolate(input=depth_unfold, scale_factor=self.up_scale, mode='nearest', align_corners=None)
        # print(depth_unfold[0,:,8,8])
        # print(depth_unfold[0,:,15,15])
        # print(depth[0,0,:3,:3])
        

        out = (depth_unfold * weight).sum(dim = 1).unsqueeze(dim = 1)

        if self.residual:
            out += F.interpolate(input=depth, scale_factor=self.up_scale, mode='nearest', align_corners=None)

        return out

class DJFR(nn.Module):
    def __init__(self, residual=True):
        super(DJFR, self).__init__()
        self.T = nn.Sequential(
            nn.Conv2d(1, 96, 9, padding=4),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(96, 48, 1, padding=0),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(48, 1, 5, padding=2),
            )

        self.G = nn.Sequential(
            nn.Conv2d(3, 96, 9, padding=4),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(96, 48, 1, padding=0),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(48, 1, 5, padding=2),
            )

        self.F = nn.Sequential(
            nn.Conv2d(2, 64, 9, padding=4),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, padding=0),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 5, padding=2),
            )

        self.residual = residual
               
    def forward(self, x):
        image, depth = x

        T = self.T(depth)
        G = self.G(image)
        F = self.F(torch.cat((T,G), dim=1))

        if self.residual:
            out = F + depth   
        else:
            out = F

        return out

class DMSG(nn.Module):
    def __init__(self, residual=True):
        super(DMSG, self).__init__()
        self.Y1 = nn.Sequential(
            nn.Conv2d(3, 49, 7, padding=3),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(49, 32, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            )

        self.Y2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            )

        self.Y3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            )

        self.D1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),
            )

        self.D2 = nn.Sequential(
            nn.Conv2d(64, 32, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),            
            nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
            )

        self.D3 = nn.Sequential(
            nn.Conv2d(64, 32, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),            
            nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
            )

        self.D4 = nn.Sequential(
            nn.Conv2d(64, 32, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),      
            nn.Conv2d(32, 32, 5, padding=2),
            # nn.BatchNorm2d((up_scale*kpn_kernel)**2),
            nn.ReLU(),         
            nn.Conv2d(32, 1, 5, padding=2),
            )
               
    def forward(self, x):
        image, depth_lr, depth_hr = x

        Y1 = self.Y1(image)
        Y2 = self.Y2(Y1)
        Y3 = self.Y3(Y2)

        D1 = self.D1(depth_lr)
        D2 = self.D2(torch.cat((D1,Y3),dim=1))
        D3 = self.D3(torch.cat((D2,Y2),dim=1))       
        D4 = self.D4(torch.cat((D3,Y1),dim=1))

      
        out = D4 + depth_hr   

        return out




