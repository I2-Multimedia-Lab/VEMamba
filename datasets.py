import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from skimage import io
import h5py
import random
import torch.nn.functional as F
import imageio
from degradation import SRMDPreprocessing

def save_tensor_as_tiff(tensor,name='test'):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    tensor = tensor.numpy()

    if tensor.dtype != np.uint8:
        tensor = (tensor * 255).astype(np.uint8)
    
    imageio.volwrite(f'./{name}.tif', tensor)
def rotate_rand8(imgs_hr):
    '''
    random orthogonal rotations, keeping anisotropic axis unchanged.
    imgs_hr should be a 3d data with (Z,Y,X) order.
    '''
    alpha = random.choice([-2, 0])
    beta = random.choice([-2])
    gamma = random.choice([-2, -1, 0, 1])

    imgs_hr_copy = imgs_hr.clone()
    imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + alpha, [2, 4])
    imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + beta, [2, 3])
    imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + gamma, [3, 4])

    return imgs_hr_copy
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class ImageDataset_train(Dataset):
    '''
    Create training dataset.
    :param image_pth: the path of volume electron microscopy 3d input data.
    :param image_split: the percent of training data in the input data.
    :param subvol_shape: the subvolume size for model processing.
    :param scale_factor: the scale factor of isotropic reconstruction.
    :param is_inpaint: if False, perform normal isovem training, learning isotropic reconstruction task.
                       if True, perform isovem+ training, co-learning the slice inpainting and isotropic reconstruction task.
    :return hr and lr: self-supervised paired data for isotropic reconstruction training.
    :return name: the 3d coordination of current subvolume.
    '''
    def __init__(self, image_pth, image_split, subvol_shape, scale_factor, is_inpaint=False):
        super(ImageDataset_train, self).__init__()

        # load data
        if image_pth.split('.')[-1]=='tif':
            self.image_file = io.imread(image_pth)
        elif image_pth.split('.')[-1] == 'h5':
            self.image_file = np.array(h5py.File(image_pth, 'r')['raw'])
        else:
            raise ValueError(f'Not support the image format of {image_pth}')

        # split train set
        shapey = self.image_file.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file=self.image_file[:,0:shapey_train,:]

        # train settings
        self.subvol_shape = subvol_shape
        self.scale_factor= scale_factor
        self.is_inpaint = is_inpaint

        # dataset size
        self.len=512 # the number of random subvolume sampling per training epoch
        
        self.degrade = SRMDPreprocessing(scale=1,blur_type='iso_gaussian',kernel_size=0,mode='bicubic',sig=0,sig_min=0,sig_max=0,noise=0)
        
    def add_degradation(self,hr):
        # hr_blur = torch.tensor(hr)
        hr_blur,_= self.degrade(hr,random=True)
        return hr_blur


    def __getitem__(self, index):
        # random crop
        z = np.random.randint(0, self.image_file.shape[0] - self.subvol_shape[0])
        y = np.random.randint(0, self.image_file.shape[1] - self.subvol_shape[1])
        x = np.random.randint(0, self.image_file.shape[2] - self.subvol_shape[2])
        img = self.image_file[z:z + self.subvol_shape[0], y:y + self.subvol_shape[1], x:x + self.subvol_shape[2]]

        # hr
        hr_transform = transforms.Compose([transforms.ToTensor()])
        img_hr = hr_transform(img.transpose(1, 2, 0).astype('uint8'))
        img_hr = img_hr.unsqueeze(0).unsqueeze(0)
        img_hr = rotate_rand8(img_hr)# 0-1
        
        # add blur
        hr_blur = self.add_degradation(img_hr*255.0)
        hr_blur = hr_blur/255.0
        
        # save_tensor_as_tiff(hr_blur[0,0,:,:,:],"blur")
        # save_tensor_as_tiff(img_hr[0,0,:,:,:],"hr")
        
        # lr
        down_sample = torch.nn.AvgPool3d(kernel_size=(1, self.scale_factor, 1))
        img_lr = down_sample(hr_blur)
        if self.is_inpaint: # random set the slice to be black
            n=np.random.randint(0, img_lr.shape[2]-1)
            img_lr[:,:,n,:,:]=0

        img_hr = img_hr.squeeze()
        img_lr = img_lr.squeeze()
        # interpolate
        img_lr = img_lr.unsqueeze(0).unsqueeze(0)
        out = F.interpolate(img_lr, size=(16, 128, 128), mode='trilinear', align_corners=False)
        out = out.squeeze(0).squeeze(0)
        img_lr = img_lr.squeeze(0).squeeze(0)

        return {"hr": img_hr,"lr": out,"lr_moco":img_lr, "subvol": (z, y, x)}


    def __len__(self):
        return self.len


class ImageDataset_val(Dataset):
    '''
    Create validation dataset
    :param image_pth: the path of volume electron microscopy 3d input data.
    :param image_split: the percent of training data in the input data.
    :param subvol_shape: the subvolume size for model processing.
    :param scale_factor: the scale factor of isotropic reconstruction.
    :param is_inpaint: if False, perform normal isovem training, learning isotropic reconstruction task.
                       if True, perform isovem+ training, co-learning the slice inpainting and isotropic reconstruction task.
    :return hr and lr: self-supervised paired data for isotropic reconstruction training.
    :return name: the 3d coordination of current subvolume.
    '''
    def __init__(self, image_pth, image_split, subvol_shape, scale_factor, is_inpaint=False):
        super(ImageDataset_val, self).__init__()

        # load data
        if image_pth.split('.')[-1]=='tif':
            self.image_file = io.imread(image_pth)
        elif image_pth.split('.')[-1] == 'h5':
            self.image_file = np.array(h5py.File(image_pth, 'r')['raw'])
        else:
            raise ValueError(f'Not support the image format of {image_pth}')

        # split val set
        shapey = self.image_file.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file=self.image_file[:,shapey_train:,:]

        # val settings
        self.subvol_shape = subvol_shape
        self.scale_factor= scale_factor
        self.is_inpaint = is_inpaint

        # generate subvolume coordinates in order, different from random cropping in training dataset
        import math
        import itertools
        z_crop_num = math.ceil(self.image_file.shape[0] / subvol_shape[0])
        y_crop_num = math.ceil(self.image_file.shape[1] / subvol_shape[1])
        x_crop_num = math.ceil(self.image_file.shape[2] / subvol_shape[2])
        self.coords= []
        for z, y, x in itertools.product(range(z_crop_num), range(y_crop_num), range(x_crop_num)):
            z_coord = z * subvol_shape[0] if z * subvol_shape[0] < self.image_file.shape[0] - subvol_shape[0] else self.image_file.shape[0] - subvol_shape[0]
            y_coord = y * subvol_shape[1] if y * subvol_shape[1] < self.image_file.shape[1] - subvol_shape[1] else self.image_file.shape[1] - subvol_shape[1]
            x_coord = x * subvol_shape[2] if x * subvol_shape[2] < self.image_file.shape[2] - subvol_shape[2] else self.image_file.shape[2] - subvol_shape[2]
            self.coords.append((z_coord,y_coord,x_coord))
        self.len = len(self.coords) # the number of cropped subvolumes
        self.degrade = SRMDPreprocessing(scale=1,blur_type='iso_gaussian',kernel_size=0,mode='bicubic',sig=0,sig_min=0,sig_max=0,noise=0.0)
    
    def add_degradation(self,hr):
        # hr_blur = torch.tensor(hr)
        hr_blur,_= self.degrade(hr,random=False)
        return hr_blur


    def __getitem__(self, index):
        # crop out subvolume based on coordinate
        z,y,x=self.coords[index]
        img = self.image_file[z:z + self.subvol_shape[0], y:y + self.subvol_shape[1], x:x + self.subvol_shape[2]]

        # hr
        hr_transform = transforms.Compose([transforms.ToTensor()])
        img_hr = hr_transform(img.transpose(1, 2, 0).astype('uint8'))
        img_hr = img_hr.unsqueeze(0).unsqueeze(0)
        
        hr_blur = self.add_degradation(img_hr*255.0)
        hr_blur = hr_blur/255.0

        # lr
        down_sample = torch.nn.AvgPool3d(kernel_size=(1, self.scale_factor, 1))
        img_lr = down_sample(hr_blur)
        if self.is_inpaint: # random set the slice to be black
            n=np.random.randint(0, img_lr.shape[2]-1)
            img_lr[:,:,n,:,:]=0

        img_hr = img_hr.squeeze()
        img_lr = img_lr.squeeze()

        # interpolate
        img_lr = img_lr.unsqueeze(0).unsqueeze(0)
        out = F.interpolate(img_lr, size=(16, 128, 128), mode='trilinear', align_corners=False)
        out = out.squeeze(0).squeeze(0)
        img_lr = img_lr.squeeze(0).squeeze(0)

        return {"hr": img_hr,"lr": out, "lr_moco":img_lr,"subvol": (z, y, x)}



    def __len__(self):
        return self.len

