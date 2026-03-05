import torch 
import torch.nn as nn
from torch.utils.data import DataLoader


from datetime import datetime
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from model.moco import MoCo,Encoder
from datasets import ImageDataset_train,ImageDataset_val



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MocoTrainer():
    def __init__(self,root_path) -> None:
        self.root_path = root_path
          
        train_dataset = ImageDataset_train(image_pth=f"{self.root_path}/datasets/x.tif",image_split=0.8,subvol_shape=(16,128,128),scale_factor=8,is_inpaint=False)
        self.train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
        test_dataset = ImageDataset_val(image_pth=f'{self.root_path}/datasets/x.tif',image_split=0.8,subvol_shape=(16,128,128),scale_factor=8,is_inpaint=False)
        test_dataset.len = 512
        self.test_dataloader = DataLoader(test_dataset,batch_size=128)
        
        self.model = MoCo(base_encoder=Encoder).cuda()
        self.contrast_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,step_size=100,gamma=0.5)
        
        
        self.epochs = 10000
        
        self.logger,_ = self._set_log(f'{self.root_path}/log')
        
    
    def train_one_epoch(self,epoch):
        losses_contrast = AverageMeter()
        self.model.train()
        for _,train_data in enumerate(self.train_dataloader):
            train_data['lr'] = train_data['lr'].cuda()
            fea, output, target = self.model(im_q = train_data['lr'][:,0:8,:,:],im_k = train_data['lr'][:,8:,:,:])
            
            self.optimizer.zero_grad()
            loss = self.contrast_loss(output,target)
        
            losses_contrast.update(loss.item())
            
            loss.backward()
            self.optimizer.step()

        self.logger.info(f"Training [{epoch}\{self.epochs}]  loss:  {losses_contrast.avg}")
        
    
    def valid(self,epoch):
        losses_contrast = AverageMeter()
        # self.model.eval()
        with torch.no_grad():
            for _,test_data in  enumerate(self.test_dataloader):
                test_data['lr'] = test_data['lr'].cuda()
                fea, output, target = self.model(im_q = test_data['lr'][:,0:8,:,:],im_k = test_data['lr'][:,8:,:,:])
                
                loss = nn.CrossEntropyLoss().cuda()(output,target)
                losses_contrast.update(loss.item())
                
        self.logger.info(f"Testing [{epoch}\{self.epochs}]  loss:  {losses_contrast.avg}")
                
    
    def train(self):
        for i in range(self.epochs):
            self.train_one_epoch(epoch=i)
            self.scheduler.step()
            if i%5 == 0:
                self.valid(epoch=i)
            if i%100==0 and i>0:
                self.save_model(epoch=i)
    
    def save_model(self,epoch):
        self.logger.info("Saving model......")
        save_path = f"{self.root_path}/checkpoints/moco"
        os.makedirs(save_path,exist_ok=True)
        torch.save(self.model.state_dict(),f"{save_path}/moco{epoch}.pth")
        
    def _set_log(self,save_dir):
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        # 生成日志文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(save_dir, f"training_{timestamp}.log")
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


if __name__=="__main__":
    trainer = MocoTrainer(root_path='/home/user/VEMamba')
    trainer.train()    
