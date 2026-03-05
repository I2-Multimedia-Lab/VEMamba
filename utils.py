import itertools
import random
import torch
import math
import numpy as np
import cv2
import ctypes, inspect
from PyQt5.QtCore import QThread, QObject, pyqtSignal
import argparse

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

"""Particle color for different categories."""
colors = [(244, 67, 54),  # 1
          (255, 235, 59),  # 2
          (156, 39, 176),  # 3
          (33, 150, 243),  # 4
          (0, 188, 212),  # 5
          (139, 195, 74),  # 6
          (255, 152, 0),  # 7
          (63, 81, 181),  # 8
          (255, 193, 7),  # 9
          (255, 0, 0),  # 10
          (0, 255, 0),  # 11
          (0, 0, 255),  # 12
          (255, 255, 0),  # 13
          (255, 0, 255),  # 14
          (0, 255, 255),  # 15
          ]

def stretch(tomo):
    tomo = (tomo - np.min(tomo)) / (np.max(tomo) - np.min(tomo)) * 255
    return np.array(tomo).astype(np.uint8)

def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

def add_transparency(img, label, alpha):
    img = np.array(255.0 * rescale(img), dtype=np.uint8)
    label = (label / label.max() * 255).astype('uint8')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    label_rgb = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)

    label_rgb[:, :, 2] = 0
    label_rgb[:, :, 1] = 0
    out = cv2.addWeighted(img_rgb, 1, label_rgb, alpha, 0)

    return np.array(out)


'''add dict to args'''
def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


'''3D Orthogonal rotations'''
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

def rotate_8(imgs_hr):
    '''
    8 kinds of orthogonal rotations, keeping anisotropic axis unchanged.
    imgs_hr should be a 3d data with (Z,Y,X) order.
    utils for test-time-augmentation.
    '''
    alpha_ls = [-2,0]
    beta_ls = [-2]
    gamma_ls = [-2, -1,0,1]

    imgs_hr_ls=[]
    idx=-1
    for alpha in alpha_ls:
        for beta in beta_ls:
            for gamma in gamma_ls:
                idx+=1
                imgs_hr_copy = imgs_hr.clone()
                imgs_hr_copy=torch.rot90(imgs_hr_copy, 2+alpha, [2, 4])
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + beta, [2, 3])
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + gamma, [3, 4])
                imgs_hr_ls.append(imgs_hr_copy)
    return imgs_hr_ls


def inv_rotate_8(imgs_hr_ls, image_shape=(128, 128, 128)):
    '''
    inverse rotations for 8 orthogonal rotations.
    images in imgs_hr_ls should be 3d data with (Z,Y,X) order.
    utils for test-time-augmentation.
    '''
    alpha_ls = [-2, 0]
    beta_ls = [-2]
    gamma_ls = [-2, -1, 0, 1]

    idx=-1
    imgs_anti_rot =[]
    res=torch.zeros((1,1,image_shape[0],image_shape[1],image_shape[2])).type(Tensor)
    for alpha in alpha_ls:
        for beta in beta_ls:
            for gamma in gamma_ls:
                idx+=1
                imgs_hr_copy = imgs_hr_ls[idx]
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 4-(2 +gamma), [3, 4])
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 4-(2 + beta), [2, 3])
                imgs_hr_copy=torch.rot90(imgs_hr_copy, 4-(2+alpha), [2, 4])
                imgs_anti_rot.append(imgs_hr_copy)
                res+=imgs_hr_copy

    return imgs_anti_rot,res/8


'''cropping subvolumes'''
def get_crop_num(length,crop_size=128,overlap=16):
    '''
    calculate the number of overlapping cropping
    length=n*crop_size-(n-1)*overlap
    '''
    num=(length-overlap)/(crop_size-overlap)
    return math.ceil(num)


def create_coord_2d(s1, s2=(128, 128), overlap=16):
    '''
    calculate the cropping coordinates to extract subvolumes
    '''
    coord_ls=[[],[]]
    y_crop_num=get_crop_num(s1[0],crop_size=s2[0],overlap=overlap)
    x_crop_num=get_crop_num(s1[1],crop_size=s2[1],overlap=overlap)

    for y,x in itertools.product(range(y_crop_num),range(x_crop_num)):
        y_coord = s2[0]//2 + y * (s2[0] - overlap) if (s2[0]//2 + y * (s2[0] - overlap)) < s1[0] - s2[0]//2 else s1[0] - s2[0]//2
        x_coord = s2[1]//2 + x * (s2[1] - overlap) if (s2[1]//2 + x * (s2[1] - overlap)) < s1[1] - s2[1]//2 else s1[1] - s2[1]//2
        coord_ls[0].append(y_coord)
        coord_ls[1].append(x_coord)

    return np.array(coord_ls),y_crop_num,x_crop_num


def create_coord_3d(s1, s2=(128, 128, 128), overlap=16):
    '''
    calculate the cropping coordinates to extract subvolumes
    '''
    coord_ls=[[],[],[]]
    z_crop_num=get_crop_num(s1[0],crop_size=s2[0],overlap=overlap)
    y_crop_num=get_crop_num(s1[1],crop_size=s2[1],overlap=overlap)
    x_crop_num=get_crop_num(s1[2],crop_size=s2[2],overlap=overlap)

    for z,y,x in itertools.product(range(z_crop_num), range(y_crop_num),range(x_crop_num)):
        z_coord = s2[0]//2 + z * (s2[0] - overlap) if (s2[0]//2 + z * (s2[0] - overlap)) < s1[0] - s2[0]//2 else s1[0] - s2[0]//2
        y_coord = s2[1]//2 + y * (s2[1] - overlap) if (s2[1]//2 + y * (s2[1] - overlap)) < s1[1] - s2[1]//2 else s1[1] - s2[1]//2
        x_coord = s2[2]//2 + x * (s2[2] - overlap) if (s2[2]//2 + x * (s2[2] - overlap)) < s1[2] - s2[2]//2 else s1[2] - s2[2]//2
        coord_ls[0].append(z_coord)
        coord_ls[1].append(y_coord)
        coord_ls[2].append(x_coord)

    return np.array(coord_ls),z_crop_num,y_crop_num,x_crop_num


'''2D stitch'''
def stitch_X(img1, img2, overlap):
    '''2d stitch along X axis.'''
    b, a = img1.shape
    d, c = img2.shape
    res = np.zeros([b, a+c - overlap]).astype(np.float16)

    for i in range(a+c - overlap):
        if i < a-overlap:
            weight1 = 1
            res[:, i] = weight1 * img1[:, i]
        if a-overlap <= i < a:
            weight1 = -i + a
            weight2 = i - a + overlap
            weight1 = weight1 / overlap
            weight2 = weight2 / overlap
            res[:, i] = weight1 * img1[:, i] + weight2 * img2[:, i - a + overlap]
        if i >= a:
            weight2 = 1
            res[:, i] = weight2 * img2[:, i-a+overlap]

    return res


def stitch_Y(img1, img2, overlap):
    '''2d stitch along Y axis.'''
    b, a = img1.shape
    d, c = img2.shape
    res = np.zeros([b+d - overlap, a]).astype(np.float16)
    for i in range(b+d - overlap):
        if i < b-overlap:
            weight1 = 1
            res[i, :] = weight1 * img1[i, :]
        if b-overlap <= i < b:
            weight1 = -i + b
            weight2 = i - b + overlap
            weight1 = weight1 / overlap
            weight2 = weight2 / overlap
            res[i, :] = weight1 * img1[i, :] + weight2 * img2[i - b + overlap, :]
        if i >= b:
            weight2 = 1
            res[i, :] = weight2 * img2[i-b+overlap, :]

    return res


def stitch2D(vol_ls,whole_shape, subvol_shape, subvol_overlap):
    '''stitch images in 2D, the overlap region is linear fused to avoid edges.'''
    y_crop_num=get_crop_num(whole_shape[0],subvol_shape[0],subvol_overlap)
    x_crop_num=get_crop_num(whole_shape[1],subvol_shape[1],subvol_overlap)

    # 2d stitch along x axis
    x_ls = []
    for j in range(y_crop_num):
        x_temp = None
        for k in range(x_crop_num):
            nps = np.s_[j * x_crop_num + k]
            if x_temp is not None:
                ovlp = subvol_overlap if x_temp.shape[1] + vol_ls[nps].shape[1] - subvol_overlap <= whole_shape[1] \
                    else x_temp.shape[1] + vol_ls[nps].shape[1] - whole_shape[1]
                x_temp = stitch_X(x_temp, vol_ls[nps], overlap=ovlp)
            else:
                x_temp = vol_ls[nps]
        x_ls.append(x_temp)

    # 2d stitch along y axis
    y_temp = None
    for j in range(y_crop_num):
        nps = np.s_[j]
        if y_temp is not None:
            ovlp = subvol_overlap if y_temp.shape[0] + x_ls[nps].shape[0] - subvol_overlap <= whole_shape[0] \
                else y_temp.shape[0] + x_ls[nps].shape[0] - whole_shape[0]
            y_temp = stitch_Y(y_temp, x_ls[nps], overlap=ovlp)
        else:
            y_temp = x_ls[nps]

    return y_temp


'''3D stitch'''
def stitch3D_X(img1, img2, overlap=16):
    '''3d stitch along X axis, utils for 3d subvolume stitching.'''
    c, b, a = img1.shape
    z, y, x = img2.shape
    res = np.zeros([c, b, a+x - overlap]).astype(np.float16)
    for j in range(b):
        res[:, j, :]=stitch_X(img1[:, j, :], img2[:, j, :], overlap=overlap)
    return res


def stitch3D_Y(img1, img2, overlap=16):
    '''3d stitch along Y axis, utils for 3d subvolume stitching.'''
    c, b, a = img1.shape
    z, y, x = img2.shape
    res = np.zeros([c, b + y - overlap, a]).astype(np.float16)
    for j in range(a):
        res[:, :, j] = stitch_X(img1[:, :, j], img2[:, :, j], overlap=overlap)
    return res


def stitch3D_Z(img1, img2, overlap=16):
    '''3d stitch along Z axis, utils for 3d subvolume stitching.'''
    c, b, a = img1.shape
    z, y, x = img2.shape
    res = np.zeros([c + z - overlap, b, a]).astype(np.float16)
    for j in range(b):
        res[:, j, :] = stitch_Y(img1[:, j, :], img2[:, j, :], overlap=overlap)
    return res


def stitch3D(vol_ls,whole_shape, subvol_shape, subvol_overlap,scale_factor):
    '''stitch images in 3D, the overlap region is linear fused to avoid edges.'''
    z_crop_num = get_crop_num(whole_shape[0],subvol_shape[0],subvol_overlap)
    y_crop_num = get_crop_num(whole_shape[1],subvol_shape[1],subvol_overlap)
    x_crop_num = get_crop_num(whole_shape[2], subvol_shape[2], subvol_overlap)

    # 3d stitch along x axis
    x_ls=[]
    for i in range(z_crop_num):
        for j in range(y_crop_num):
            x_temp = None
            for k in range(x_crop_num):
                nps=np.s_[i * y_crop_num * x_crop_num + j * x_crop_num + k]
                if x_temp is not None:
                    ovlp=subvol_overlap if x_temp.shape[2]+vol_ls[nps].shape[2]-subvol_overlap<=whole_shape[2] \
                        else x_temp.shape[2]+vol_ls[nps].shape[2]-whole_shape[2]
                    x_temp = stitch3D_X(x_temp, vol_ls[nps], overlap=ovlp)
                else:
                    x_temp = vol_ls[nps]
            x_ls.append(x_temp)

    # 3d stitch along y axis
    y_ls=[]
    for i in range(z_crop_num):
        y_temp = None
        for j in range(y_crop_num):
            nps =np.s_[i * y_crop_num + j]
            if y_temp is not None:
                ovlp = subvol_overlap if y_temp.shape[1] + x_ls[nps].shape[1] - subvol_overlap <= whole_shape[1] \
                    else y_temp.shape[1] + x_ls[nps].shape[1] - whole_shape[1]
                y_temp = stitch3D_Y(y_temp, x_ls[nps], overlap=ovlp)
            else:
                y_temp = x_ls[nps]
        y_ls.append(y_temp)

    # 3d stitch along z axis
    z_temp=None
    for i in range(z_crop_num):
        if z_temp is not None:
            ovlp = int(subvol_overlap* scale_factor) if z_temp.shape[0] + y_ls[i].shape[0] - int(subvol_overlap* scale_factor) <= int(whole_shape[0] * scale_factor) \
                else z_temp.shape[0] + y_ls[i].shape[0] - int(whole_shape[0] * scale_factor)
            z_temp = stitch3D_Z(z_temp, y_ls[i], overlap=ovlp)
        else:
            z_temp = y_ls[i]

    return z_temp

'''convert image type'''
def float2uint8(img):
    '''convert image from float to uint8 data type.'''
    img[img < 0] = 0
    img[img > 1] = 1
    img=(img * 255).astype('uint8')
    return img

def _async_raise(tid, exctype):
    """
    stop thread
    """
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)

class EmittingStr(QObject):
    textWritten = pyqtSignal(str)

    def __init__(self):
        super(EmittingStr, self).__init__()

    def write(self, text):
        try:
            if len(str(text)) >= 2:
                self.textWritten.emit(str(text))
        except:
            pass

    def flush(self, text=None):
        pass