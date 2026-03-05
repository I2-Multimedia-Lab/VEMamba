import os
import numpy as np
from skimage import io
import torch
import torch.nn as nn
from pytorch_msssim import ssim,ms_ssim
import math
import json

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def compute_ssim(arr1, arr2,need_2d=True):
    '''ssim calculation on 3D, XY, XZ, YZ'''
    ssim_ls=[]
    ssim_3d = ssim(arr1, arr2,win_size=11,data_range=1.0)
    ssim_ls.append(ssim_3d.item())
    if need_2d:
        ssim_xy,ssim_xz,ssim_yz=0,0,0
        for i in range(arr1.shape[2]):
            ssim_xy += ssim(arr1[:,:,i, :, :], arr2[:,:,i, :, :],win_size=11,data_range=1)
        for j in range(arr1.shape[3]):
            ssim_xz += ssim(arr1[:,:,:, j, :], arr2[:,:,:, j, :],win_size=11,data_range=1)
        for k in range(arr1.shape[4]):
            ssim_yz += ssim(arr1[:,:,:, :, k], arr2[:,:,:, :, k],win_size=11,data_range=1)
        ssim_xy,ssim_xz,ssim_yz =ssim_xy/arr1.shape[2],ssim_xz / arr1.shape[3],ssim_yz / arr1.shape[4]
        ssim_ls.extend([ssim_xy.item(),ssim_xz.item(),ssim_yz.item()])

    return ssim_ls

def compute_ms_ssim(arr1, arr2,need_2d=True):
    '''ms-ssim calculation on 3D, XY, XZ, YZ'''
    ms_ssim_ls=[]
    ms_ssim_3d = ms_ssim(arr1, arr2,win_size=5, data_range=1)
    ms_ssim_ls.append(ms_ssim_3d.item())
    if need_2d:
        ms_ssim_xy,ms_ssim_xz,ms_ssim_yz=0,0,0
        for i in range(arr1.shape[2]):
            ms_ssim_xy += ms_ssim(arr1[:,:,i, :, :], arr2[:,:,i, :, :],win_size=5, data_range=1)
        for j in range(arr1.shape[3]):
            ms_ssim_xz += ms_ssim(arr1[:,:,:, j, :], arr2[:,:,:, j, :],win_size=5, data_range=1)
        for k in range(arr1.shape[4]):
            ms_ssim_yz += ms_ssim(arr1[:,:,:, :, k], arr2[:,:,:, :, k],win_size=5, data_range=1)
        ms_ssim_xy,ms_ssim_xz,ms_ssim_yz =ms_ssim_xy/arr1.shape[2],ms_ssim_xz / arr1.shape[3],ms_ssim_yz / arr1.shape[4]
        ms_ssim_ls.extend([ms_ssim_xy.item(),ms_ssim_xz.item(),ms_ssim_yz.item()])

    return ms_ssim_ls


def compute_psnr(arr1, arr2,need_2d=True):
    '''psnr calculation on 3D, XY, XZ, YZ'''
    psnr_ls=[]
    mse_3d=nn.MSELoss()(arr1, arr2)
    psnr_3d = 20 * math.log10(1 / math.sqrt(mse_3d.item()))
    psnr_ls.append(psnr_3d)
    if need_2d:
        psnr_xy,psnr_xz,psnr_yz=0,0,0
        for i in range(arr1.shape[2]):
            mse_xy = nn.MSELoss()(arr1[:,:,i, :, :], arr2[:,:,i, :, :])
            psnr_xy +=  20 * math.log10(1 / math.sqrt(mse_xy.item()))
        for j in range(arr1.shape[3]):
            mse_xz = nn.MSELoss()(arr1[:, :, :, j, :], arr2[:, :, :, j, :])
            psnr_xz += 20 * math.log10(1 / math.sqrt(mse_xz.item()))
        for k in range(arr1.shape[4]):
            mse_yz = nn.MSELoss()(arr1[:, :, : ,:, k], arr2[:, :, :, :, k])
            psnr_yz += 20 * math.log10(1 / math.sqrt(mse_yz.item()))
        psnr_xy,psnr_xz,psnr_yz =psnr_xy/arr1.shape[2],psnr_xz / arr1.shape[3],psnr_yz / arr1.shape[4]
        psnr_ls.extend([psnr_xy,psnr_xz,psnr_yz])

    return psnr_ls

def compute_lpips(arr1, arr2, need_2d=True):
    '''
    lpips calculation on XY, XZ, YZ
    Usage: conda install piq -c photosynthesis-team -c conda-forge -c PyTorch
    '''
    from piq import LPIPS
    arr1 = arr1.to(torch.float32)
    arr2 = arr2.to(torch.float32)
    loss = LPIPS()

    if need_2d:
        lpips_ls = []
        lpips_xy, lpips_xz, lpips_yz = 0, 0, 0
        for i in range(arr1.shape[2]):
            lpips_xy += loss(arr1[:, :, i, :, :], arr2[:, :, i, :, :])
        # for j in range(arr1.shape[3]):
        #     lpips_xz += loss(arr1[:, :, :, j, :], arr2[:, :, :, j, :])
        # for k in range(arr1.shape[4]):
        #     lpips_yz += loss(arr1[:, :, :, :, k], arr2[:, :, :, :, k])
        # lpips_xy, lpips_xz, lpips_yz = lpips_xy / arr1.shape[2], lpips_xz / arr1.shape[3], lpips_yz / arr1.shape[4]
        lpips_xy = lpips_xy / arr1.shape[2]
        lpips_ls.extend([lpips_xy.item()
                         # , lpips_xz.item(), lpips_yz.item()
                         ])
    return lpips_ls


def error_map(arr1, arr2,save_dir):
    '''visualize the error map'''
    err=torch.abs(arr1-arr2)
    err_np=np.array(err.squeeze()*255).astype('uint8')
    io.imsave(os.path.join(save_dir,'error_map.tif'),err_np)


def calculate_metrics(arr1,arr2,save_json=None,is_cuda=False,vis_error=False):
    '''calculate performance metrics, and save to json.'''
    assert arr1.shape == arr2.shape
    arr1 = torch.tensor(arr1[np.newaxis, np.newaxis, ...]/255.0)
    arr2 = torch.tensor(arr2[np.newaxis, np.newaxis, ...]/255.0)
    if is_cuda:
        arr1=arr1.cuda()
        arr2=arr2.cuda()

    metrics={}
    metrics['ssim']=compute_ssim(arr1, arr2)
    print('ssim done')
    metrics['ms_ssim'] = compute_ms_ssim(arr1, arr2)
    print('ms-ssim done')
    metrics['psnr'] = compute_psnr(arr1, arr2)
    print('psnr done')
    metrics['lpips'] = compute_lpips(arr1, arr2)
    print('lpips done')

    if vis_error:
        error_dir=os.path.dirname(save_json)
        error_map(arr1, arr2,error_dir)

    if save_json:
        with open(save_json, "w") as f:
            f.write(json.dumps(metrics, ensure_ascii=False, indent=4, separators=(',', ':')))

    return metrics