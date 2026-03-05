import argparse
from datasets import *
from metrics import *
from utils import *
from tqdm import tqdm
import sys

def run_model_isosr(model,moco,image,scale_factor,args):

    coord_np,z_crop_num,y_crop_num,x_crop_num = create_coord_3d(image.shape, args.test_shape, args.test_overlap)
    vol_pred_ls=[]
    vol_uncertainty_ls = []

    with torch.no_grad():
        # for each subvolume
        for i in tqdm(range(coord_np.shape[1]), desc='IsoSR'):
            z, y, x = coord_np[0, i], coord_np[1, i], coord_np[2, i]
            crop = np.s_[z - args.test_shape[0] // 2:z + args.test_shape[0] // 2, y - args.test_shape[1] // 2:y + args.test_shape[1] // 2,
                   x - args.test_shape[2] // 2:x + args.test_shape[2] // 2]
            batch = image[crop]
            # eight rotations
            batch = torch.tensor(batch[np.newaxis, np.newaxis, ...] / 255.0).type(Tensor)
            batch_rot_ls = rotate_8(batch)
            pred_rot_ls = []
            # for each rotated subvolume
            for j in range(0, len(batch_rot_ls)):
                batch_rot = batch_rot_ls[j]
                cdp = moco(batch_rot[:,0,0:8,:,:],batch_rot[:,0,8:,:,:])
                # model inference
                batch_rot=batch_rot.squeeze().permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
                pred_rot = model(batch_rot,cdp)
                pred_rot =pred_rot.squeeze().permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
                pred_rot_ls.append(pred_rot)
            # save isosr
            pred_ls, pred = inv_rotate_8(pred_rot_ls, (pred_rot.shape[2], pred_rot.shape[3], pred_rot.shape[4]))
            
            pred = pred.squeeze().cpu().detach().numpy()

            vol_pred_ls.append(pred)

    # ----------
    #  Free Memory
    # ----------
    del pred_rot
    del batch_rot
    del pred_rot_ls
    del batch_rot_ls
    del pred
    del batch
    torch.cuda.empty_cache()

    # ----------
    #  3D stitch
    # ----------
    print("3D Stitching takes some time which running on CPU...")
    stitch_isosr = stitch3D(vol_pred_ls, image.shape, args.test_shape, args.test_overlap, args.test_upscale)
    print("Reconstructed image has stitched.")
    
    return float2uint8(stitch_isosr)

def test_func(args):

    image_file = io.imread(args.test_data_pth)
    # image_file = image_file[0:165//args.test_upscale,0:768,0:1024]
    image_file = image_file[0:2170,0:170,0:2170]
    device=torch.device("cuda")
    from model.vemamba import VEMamba
    from model.moco import MoCo,Encoder


    moco = MoCo(base_encoder=Encoder).cuda()
    moco.load_state_dict(torch.load('model.pth'),strict=False)
    moco.eval()
    model = VEMamba(input_resolution=(16,128),upscales=args.test_upscale).cuda()
    model.load_state_dict(torch.load(args.test_ckpt_path),strict=False)
    model.eval()
    

    args.test_shape = (16, 128, 128)
    args.test_overlap = 8

    pred_isosr = run_model_isosr(model,moco, image_file, args.test_upscale, args)
    pred_isor_transposed = np.transpose(pred_isosr, (1, 0, 2))
    pred_isor_transposed = pred_isor_transposed[38:163,911:2161,911:2161]
    io.imsave(os.path.join(args.test_output_dir, 'isosr.tif'), pred_isor_transposed)
    # io.imsave(os.path.join(args.test_output_dir, 'ours_4x_cremi_epfl.tif'), pred_isosr)
    print('Isotropic reconstructed volume is saved to {}, named {}'.format(args.test_output_dir, 'isosr.tif'))

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser(description='Parameters for VEMamba Testing')
    parser.add_argument('--test_config_path', help='path of test config file', type=str,
                        default="config/test_4x.json")

    with open(parser.parse_args().test_config_path, 'r', encoding='UTF-8') as f:
        test_config = json.load(f)
    add_dict_to_argparser(parser, test_config)
    args = parser.parse_args()
    print(args)

    test_func(args)
