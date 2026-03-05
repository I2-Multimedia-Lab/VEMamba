import math
from torch.optim.lr_scheduler import LambdaLR
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from datetime import datetime
from tqdm import tqdm
import logging
import argparse



from model.moco import MoCo,Encoder
from dataset_cremi import ImageDataset_train,ImageDataset_val
from metrics import *

from model.vemamba import VEMamba


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function after
    a linear warmup period from 0 to 1.
    """
    def lr_lambda(current_step):
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
class Trainer():
    def __init__(self,root_path,arg) -> None:
        self.root_path = root_path
        self.arg = arg
          
        train_dataset = ImageDataset_train(image_pth=os.path.join(self.root_path,self.arg['train_data_path']),
                                           image_split=self.arg['train_data_splits'],
                                           subvol_shape=self.arg['train_subvol_shape'],
                                           scale_factor=self.arg['train_upscale'],
                                           is_inpaint=False)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.arg['train_batch_size'],
                                           shuffle=True)
        test_dataset = ImageDataset_val(image_pth=os.path.join(self.root_path,self.arg['train_data_path']),
                                        image_split=self.arg['train_data_splits'],
                                        subvol_shape=self.arg['train_subvol_shape'],
                                        scale_factor=self.arg['train_upscale'],
                                        is_inpaint=False)
        
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.arg['train_batch_size'])
        # model = nn.DataParallel(model, device_ids=[0, 1])# 先指定双卡，再CUDA（）
        
        self.model = VEMamba(input_resolution=self.arg['input_resolution'],upscales = self.arg['train_upscale']).cuda()
        
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.arg['train_lr'],betas=(0.9,0.99))
        # self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,step_size=100,gamma=0.5)

        warmup_epochs = self.arg['train_warmup_epochs']
        total_epochs = self.arg['train_num_epochs']
        num_training_steps = len(self.train_dataloader) * total_epochs
        num_warmup_steps = len(self.train_dataloader) * warmup_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )



        self.epochs = self.arg['train_num_epochs']
        
        self.logger,_ = self._set_log(os.path.join(self.root_path,self.arg['train_log_path']))
        self.logger.info(f"Training Configuration: {self.arg}")
        self._load_moco()

    def _load_moco(self):
        moco = MoCo(base_encoder=Encoder).cuda()
        
        moco.load_state_dict(torch.load(os.path.join(self.root_path,self.arg['moco_checkpoint_path'])),strict=False)
        moco.eval()
        self.moco = moco
    
    def _set_log(self,save_dir):
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        # 生成日志文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(save_dir, f"training_vem_{timestamp}.log")
        # 创建logger
        logger = logging.getLogger("PyTorch_Training")
        logger.setLevel(logging.INFO)
        # 清除现有处理器（避免重复日志）
        if logger.hasHandlers():
            logger.handlers.clear()
        # 文件处理器（写入日志文件）
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        # 控制台处理器（输出到终端）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger, log_file
        
    
    def train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0.0
        total_l1_loss = 0.0
        total_ssim_loss = 0.0
        for iter, batch in enumerate(tqdm(self.train_dataloader, desc="Training"), 1):
            self.optimizer.zero_grad()
            # batch['lr'] = batch['lr'].cuda()
            batch['lr_moco'] = batch['lr_moco'].cuda()
            with torch.no_grad():
                cdp = self.moco(batch['lr_moco'][:,:8,:,:],batch['lr_moco'][:,8:,:,:])

            output = self.model(batch['lr_moco'].unsqueeze(1),cdp)
            loss_l1 = nn.L1Loss()(output,batch['hr'].unsqueeze(1).cuda())
            loss_ssim = 1 - ssim(output,batch['hr'].unsqueeze(1).cuda(),data_range=1.0,size_average=True)
            loss_lpips = compute_lpips(batch['hr'].unsqueeze(1).cuda(), output, need_2d=True)[0]
            loss = loss_l1 + loss_ssim+ loss_lpips
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            total_l1_loss += loss_l1.item()
            total_ssim_loss += loss_ssim.item()
            
        self.logger.info(f"Training [{epoch}\{self.epochs}]   loss:  {total_loss:.4f}  loss_l1: {total_l1_loss:.4f}  loss_ssim: {total_ssim_loss:.4f}")
        
    
    def valid(self,epoch):
        self.model.eval()
        val_metirc_ls = [ 0, 0, 0]
        with torch.no_grad():
            for iter, batch in enumerate(tqdm(self.test_dataloader, desc="Validation"), 1):
                # batch['lr'] = batch['lr'].cuda()
                batch['lr_moco'] = batch['lr_moco'].cuda()
                cdp = self.moco(batch['lr_moco'][:,:8,:,:],batch['lr_moco'][:,8:,:,:])
                output = self.model(batch['lr_moco'].unsqueeze(1),cdp)
                batch['hr'] = batch['hr'].unsqueeze(1).cuda()

                val_ssim = compute_ssim(batch['hr'], output, need_2d=False)[0]
                val_psnr = compute_psnr(batch['hr'], output, need_2d=False)[0]
                val_lpips = compute_lpips(batch['hr'], output, need_2d=True)[0]


                val_metirc_ls[0] += val_ssim
                val_metirc_ls[1] += val_psnr
                val_metirc_ls[2] += val_lpips

                if iter  == 1:
                    image_savedir =os.path.join(self.root_path,self.arg['train_visual_path'], "%04d" % epoch + '.tif')
                    output_np = output[0].squeeze().float().cpu().clamp_(0, 1).numpy()
                    io.imsave(image_savedir,(output_np*255).astype('uint8'))


        val_metirc_ls = [x / len(self.test_dataloader) for x in val_metirc_ls]

                
        self.logger.info(f"Validation [{epoch}\{self.epochs}]   SSIM:  {val_metirc_ls[0]:.6f}  PSNR: {val_metirc_ls[1]:.6f}   LPIPS: {val_metirc_ls[2]:.6f}")

    
    def train(self):
        for i in range(1,self.epochs+1):
            self.train_one_epoch(i)
            self.valid(i)
            self.save_model(i)

    def save_model(self,epoch):
        os.makedirs(f'{self.root_path}/checkpoints/vemamba',exist_ok=True)
        save_path = os.path.join(self.root_path,self.arg['train_checkpoint_path'])
        torch.save(self.model.state_dict(),f'{save_path}/vemamba_epoch{epoch}.pth')
        self.logger.info(f"Model saved !")
        

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser(description='Parameters for VEMamba Training')
    parser.add_argument('--train_config_path', help='path of train config file', type=str,
                        default="config/train_4x_cremi.json")
    with open(parser.parse_args().train_config_path, 'r', encoding='UTF-8') as f:
        train_config = json.load(f)
    trainer = Trainer(root_path='/home/user/VEMamba',arg=train_config)
    trainer.train()
