import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from skimage import io
import h5py
from utils import rotate_rand8
import random
from degradation import SRMDPreprocessing

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
            self.image_file1 = io.imread(image_pth)
        elif image_pth.split('.')[-1] == 'h5':
            self.image_file = np.array(h5py.File(image_pth, 'r')['raw'])
        else:
            raise ValueError(f'Not support the image format of {image_pth}')

        # split train set
        shapey = self.image_file1.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file1=self.image_file1[:,0:shapey_train,:]
        
        self.image_file2 = io.imread(image_pth.replace('A','B'))
        shapey = self.image_file2.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file2=self.image_file2[:,0:shapey_train,:]

        self.image_file3 = io.imread(image_pth.replace('A','C'))
        shapey = self.image_file3.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file3=self.image_file3[:,0:shapey_train,:]

        # train settings
        self.subvol_shape = subvol_shape
        self.scale_factor= scale_factor
        self.is_inpaint = is_inpaint

        # dataset size
        self.len=500 # the number of random subvolume sampling per training epoch

        self.degrade = SRMDPreprocessing(scale=1,blur_type='iso_gaussian',kernel_size=0,mode='bicubic',sig=0,sig_min=0,sig_max=0,noise=0)
    
    def add_degradation(self,hr):
        # hr_blur = torch.tensor(hr)
        hr_blur,_= self.degrade(hr,random=True)
        return hr_blur


    def __getitem__(self, index):
        # random crop
        z = np.random.randint(0, self.image_file1.shape[0] - self.subvol_shape[0])
        y = np.random.randint(0, self.image_file1.shape[1] - self.subvol_shape[1])
        x = np.random.randint(0, self.image_file1.shape[2] - self.subvol_shape[2])

        choice = random.choice([self.image_file1, self.image_file2, self.image_file3])

        img = choice[z:z + self.subvol_shape[0], y:y + self.subvol_shape[1], x:x + self.subvol_shape[2]]

        # hr
        hr_transform = transforms.Compose([transforms.ToTensor()])
        img_hr = hr_transform(img.transpose(1, 2, 0).astype('uint8'))
        img_hr = img_hr.unsqueeze(0).unsqueeze(0)
        img_hr = rotate_rand8(img_hr)

        # add blur
        hr_blur = self.add_degradation(img_hr*255.0)
        hr_blur = hr_blur/255.0

        # lr
        down_sample = torch.nn.AvgPool3d(kernel_size=(1, self.scale_factor, 1))
        img_lr = down_sample(img_hr)
        if self.is_inpaint: # random set the slice to be black
            n=np.random.random_integers(0, img_lr.shape[2]-1)
            img_lr[:,:,n,:,:]=0

        img_hr = img_hr.squeeze()
        img_lr = img_lr.squeeze()
        return {"hr": img_hr,"lr_moco": img_lr, "subvol": (z, y, x)}


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
            self.image_file1 = io.imread(image_pth)
        elif image_pth.split('.')[-1] == 'h5':
            self.image_file = np.array(h5py.File(image_pth, 'r')['raw'])
        else:
            raise ValueError(f'Not support the image format of {image_pth}')

        # split val set
        shapey = self.image_file1.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file1=self.image_file1[:,shapey_train:,:]

        self.image_file2 = io.imread(image_pth.replace('A','B'))
        shapey = self.image_file2.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file2=self.image_file2[:,shapey_train:,:]

        self.image_file3 = io.imread(image_pth.replace('A','C'))
        shapey = self.image_file3.shape[1]
        shapey_train = int(shapey*image_split)
        self.image_file3=self.image_file3[:,shapey_train:,:]


        # val settings
        self.subvol_shape = subvol_shape
        self.scale_factor= scale_factor
        self.is_inpaint = is_inpaint

        # generate subvolume coordinates in order, different from random cropping in training dataset
        import math
        import itertools
        z_crop_num = math.ceil((self.image_file1.shape[0]/3) / subvol_shape[0])
        y_crop_num = math.ceil((self.image_file1.shape[1]/3) / subvol_shape[1])
        x_crop_num = math.ceil((self.image_file1.shape[2]/3) / subvol_shape[2])
        self.coords= []
        for z, y, x in itertools.product(range(z_crop_num), range(y_crop_num), range(x_crop_num)):
            z_coord = z * subvol_shape[0] if z * subvol_shape[0] < self.image_file1.shape[0] - subvol_shape[0] else self.image_file1.shape[0] - subvol_shape[0]
            y_coord = y * subvol_shape[1] if y * subvol_shape[1] < self.image_file1.shape[1] - subvol_shape[1] else self.image_file1.shape[1] - subvol_shape[1]
            x_coord = x * subvol_shape[2] if x * subvol_shape[2] < self.image_file1.shape[2] - subvol_shape[2] else self.image_file1.shape[2] - subvol_shape[2]
            self.coords.append((z_coord,y_coord,x_coord))
        self.len = len(self.coords)*3 # the number of cropped subvolumes

        self.degrade = SRMDPreprocessing(scale=1,blur_type='iso_gaussian',kernel_size=0,mode='bicubic',sig=0,sig_min=0,sig_max=0,noise=0.0)
    
    def add_degradation(self,hr):
        # hr_blur = torch.tensor(hr)
        hr_blur,_= self.degrade(hr,random=False)
        return hr_blur


    def __getitem__(self, index):
        # crop out subvolume based on coordinate
        length = len(self.coords)
        if (index+1) <= length:
            z,y,x=self.coords[index]
            img = self.image_file1[z:z + self.subvol_shape[0], y:y + self.subvol_shape[1], x:x + self.subvol_shape[2]]
        elif (index+1) <= length*2:
            z,y,x=self.coords[index-length]
            img = self.image_file2[z:z + self.subvol_shape[0], y:y + self.subvol_shape[1], x:x + self.subvol_shape[2]]
        else:
            z,y,x=self.coords[index-length*2]
            img = self.image_file3[z:z + self.subvol_shape[0], y:y + self.subvol_shape[1], x:x + self.subvol_shape[2]]

        # hr
        hr_transform = transforms.Compose([transforms.ToTensor()])
        img_hr = hr_transform(img.transpose(1, 2, 0).astype('uint8'))
        img_hr = img_hr.unsqueeze(0).unsqueeze(0)

        # add blur
        hr_blur = self.add_degradation(img_hr*255.0)
        hr_blur = hr_blur/255.0

        # lr
        down_sample = torch.nn.AvgPool3d(kernel_size=(1, self.scale_factor, 1))
        img_lr = down_sample(img_hr)
        if self.is_inpaint: # random set the slice to be black
            n=np.random.random_integers(0, img_lr.shape[2]-1)
            img_lr[:,:,n,:,:]=0

        img_hr = img_hr.squeeze()
        img_lr = img_lr.squeeze()

        return {"hr": img_hr, "lr_moco": img_lr, "subvol": (z, y, x)}

    def __len__(self):
        return self.len
    
if __name__ =="__main__":
    dataset = ImageDataset_val(image_pth="/mnt/ssd2/glm/VEMamba/datasets/CREMI_Dataset_A_padded.tif",image_split=0.8,subvol_shape=(16,128,128),scale_factor=4,is_inpaint=False) # scale_factor=10 ,subvol_shape=(16,160,160)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=10)
    for i in dataloader:
        print(i)