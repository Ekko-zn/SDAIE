from config import opt
import os
local_rank = int(os.environ["LOCAL_RANK"])
# local_rank = 0
import torch

torch.cuda.set_device(local_rank)
import torch.distributed as dist
dist.init_process_group(backend="nccl")
device = torch.device("cuda", local_rank)

from utils import *
from Classifier import base
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time


MSE = torch.nn.MSELoss()
CLE = torch.nn.CrossEntropyLoss()

if local_rank == 0 and opt.bc_log_save == True:
    message = print_options(opt)
    with open('log/train_detector_{}.log'.format(opt.tag),'a+') as f:
        f.write('{} \n'.format(message))

def loadpth(model, path=opt.backbone_path): 
    checkpoint = path
    netparameters = {}
    for k , v in torch.load(checkpoint,map_location='cpu')['state_net'].items():
        if 'backbone' in k:
            n = k.replace('module.backbone.', '')
            netparameters[n] = v
    model.load_state_dict(netparameters,strict=True)



class combonet(torch.nn.Module):
    def __init__(self):
        super(combonet, self).__init__()
        self.diffusionextractor_pretrained =  base()
        loadpth(self.diffusionextractor_pretrained)
        self.diffusionextractor_random =  base()
        # loadpth(self.diffusionextractor_random)

        self.fc = torch.nn.Linear(528*1,2)
    def forward(self,x):
        
        with torch.no_grad():
            f_pretrained,f1_pretrained,f2_pretrained,f3_pretrained,f4_pretrained,f5_pretrained = self.diffusionextractor_pretrained(x)
        f_random,f1,f2,f3,f4,f5 = self.diffusionextractor_random(x)

        out = self.fc(f_random)
        dis = MSE(f1,f1_pretrained) + MSE(f2,f2_pretrained) + MSE(f3,f3_pretrained) + 1*MSE(f4,f4_pretrained) + 1*MSE(f5,f5_pretrained)
        return out,dis



CombnNet = combonet()
optimizer = torch.optim.Adam(CombnNet.parameters(),lr=opt.lr)



CombnNet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(CombnNet).to(device)
CombnNet = torch.nn.parallel.DistributedDataParallel(CombnNet, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
# CombnNet = CombnNet.cuda()



# 



TrainDataset = DatasetforDiffusionGranularityBinary(opt.bc_trainset_path,flag='train',patch_type = 'fix')

train_sampler = torch.utils.data.distributed.DistributedSampler(TrainDataset,shuffle=True)
TrainDataloader = torch.utils.data.DataLoader(TrainDataset,
                                              batch_size=100,
                                              num_workers=7,
                                              sampler=train_sampler,
                                              drop_last = True)


def train(epoch):
    CombnNet.train()
    Loss1 = []
    Loss_d = []
    y_true1, Y_pred1 = [], []
    correct = 0
    count = 0 
    idx = 0
    start = time.time()
    for imgs_diffusion,labels in TrainDataloader:
        idx += 1
        imgs_diffusion, labels =  imgs_diffusion.to(device), labels.to(device).to(dtype=torch.long)
        out1,dis  = CombnNet(imgs_diffusion)
        y_pred1 = torch.argmax(out1,dim=1)
        loss1 = CLE(out1,labels)
        loss = loss1  + opt.alpha*dis
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss_d.append(dis.item())
        Loss1.append(loss1.item())
        y_true1.append(labels.cpu())
        Y_pred1.append(y_pred1.cpu())
        correct += torch.sum(y_pred1==labels)
        count += len(imgs_diffusion)
        acc = correct / count
        if local_rank == 0:
            print('Train | epoch:{} | finished:{:.4f} | acc:{:.4f} | dis:{:.4f} |cost:{:.4f} '.format(epoch,idx/len(TrainDataloader),100*acc,np.mean(Loss_d),time.time()-start),end='\r')
    if local_rank == 0 and opt.bc_log_save == True:
        with open('log/train_detector_{}.log'.format(opt.tag),'a+') as f:
            f.write('Train | epoch:{} | finished:{:.4f} | acc:{:.4f} | dis:{:.4f} \n'.format(epoch,idx/len(TrainDataloader),100*acc,np.mean(Loss_d)))
            
            

# This test function is for quick validation only and uses 16 random patches per image. For final results, use bc_eval.py; it processes the full image using non-overlapping patches, which typically yields higher detection accuracy.
def test(epoch):
    CombnNet.eval()
    aver_acc = []
    vals = ['progan','DALLE2','stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
        'stylegan2', 'whichfaceisreal',
        'ADM','Glide','Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong']
    for test_id in vals:
        Loss1 = []
        y_true1, Y_pred1 = [], []
        TestDataset = DatasetforDiffusionGranularityBinary('{}/{}/'.format(opt.test_image_path,test_id),flag='test',patch_type = 'fix')
        TestDataloader = torch.utils.data.DataLoader(TestDataset,
                                                    batch_size=100,
                                                    shuffle=True,
                                                    num_workers=7,
                                                    drop_last = True)
        count = 0
        correct = 0
        for (imgs_diffusion,labels) in (TestDataloader):
            imgs_diffusion, labels = imgs_diffusion.to(device), labels.to(device).to(dtype=torch.long)
            out1,_  = CombnNet(imgs_diffusion)
            y_pred1 = torch.argmax(out1,dim=1)
            loss1 = CLE(out1,labels)
            loss = loss1*1 
            Loss1.append(loss1.item())
            y_true1.append(labels.cpu())
            Y_pred1.append(y_pred1.cpu())
            correct += torch.sum(y_pred1==labels)
            count += len(imgs_diffusion)
            acc = correct / count
        y_true1,Y_pred1 = torch.stack(y_true1).view(-1,1), torch.stack(Y_pred1).view(-1,1)
        print('****** test:{} | Loss:{:.2f} | acc:{:.2f}'.format(test_id, np.mean(Loss1),100*acc))
        if opt.bc_log_save == True:
            with open('log/train_detector_{}.log'.format(opt.tag),'a+') as f:
                f.write('****** test:{} | Loss:{:.2f} | acc:{:.2f}\n'.format(test_id, np.mean(Loss1),100*acc))
        aver_acc.append(acc.item())
    print('Test | epoch :{} | average:{}'.format(epoch, np.mean(aver_acc)))
    if opt.bc_log_save == True:
        with open('log/train_detector_{}.log'.format(opt.tag),'a+') as f:
            f.write('Test | epoch :{} | average:{}\n'.format(epoch, np.mean(aver_acc)))

if __name__ == '__main__':
    start_epoch = 0
    for epoch in range(start_epoch+1,16):
        train_sampler.set_epoch(epoch)
        train(epoch)
        if epoch % 5 == 0:
            with torch.no_grad():
                if local_rank == 0:
                    test(epoch)
        if epoch % 1 == 0:
            state_net = CombnNet.state_dict()
            state_optimizer = optimizer.state_dict()
            state = {
                'state_net':state_net,
                'state_optimizer':state_optimizer,
                'epoch':epoch
                    }
            save_dir = './ckpt/bc/{}/'.format(opt.tag)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state, save_dir+'{}.pth'.format(epoch))


