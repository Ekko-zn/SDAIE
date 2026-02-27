import os
import torch
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
import torch.distributed as dist
dist.init_process_group(backend="nccl")
device = torch.device("cuda", local_rank)
from utils import *
from Classifier import base
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



device = 'cuda'
CLE = torch.nn.CrossEntropyLoss()
BCE = torch.nn.BCELoss()

TrainDataset = DatasetforDiffusionGranularityForEXIF(flag='train')

train_sampler = torch.utils.data.distributed.DistributedSampler(TrainDataset)

TrainDataloader = torch.utils.data.DataLoader(TrainDataset,
                                              batch_size=64,
                                              shuffle = False,
                                              num_workers=6,
                                              sampler=train_sampler,
                                              drop_last = True
                                              )


rank = [0,2,3,4,7,8,12]

cls = {'1':7,'5':6,'6':4,'9':6,'10':4,'11':2,'13':5}


class diffextractor(torch.nn.Module):
    def __init__(self):
        super(diffextractor,self).__init__()
        self.backbone = base()
        self.fc1 = torch.nn.Linear(528,1)
        self.fc2 = torch.nn.Linear(528,7)
        self.fc3 = torch.nn.Linear(528,1)
        self.fc4 = torch.nn.Linear(528,1)
        self.fc5 = torch.nn.Linear(528,1)
        self.fc6 = torch.nn.Linear(528,6)
        self.fc7 = torch.nn.Linear(528,4)
        self.fc8 = torch.nn.Linear(528,1)
        self.fc9 = torch.nn.Linear(528,1)
        self.fc10 = torch.nn.Linear(528,6)
        self.fc11 = torch.nn.Linear(528,4)
        self.fc12 = torch.nn.Linear(528,2)
        self.fc13 = torch.nn.Linear(528,1)
        self.fc14 = torch.nn.Linear(528,5)


    def _single(self,x):
        f,f1,f2,f3,f4,f5 = self.backbone(x)
        out1 = self.fc1(f)
        out2 = self.fc2(f) #
        out3 = self.fc3(f)
        out4 = self.fc4(f)
        out5 = self.fc5(f)
        out6 = self.fc6(f) #
        out7 = self.fc7(f) #
        out8 = self.fc8(f)
        out9 = self.fc9(f)
        out10 = self.fc10(f) #
        out11 = self.fc11(f) #
        out12 = self.fc12(f) #
        out13 = self.fc13(f)
        out14 = self.fc14(f) #
        return [out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,out13,out14]

    def forward(self,x1,x2):
        res = []
        F_1 = self._single(x1)
        F_2 = self._single(x2)
        for item in rank:
            res.append(F_1[item] - F_2[item])
        constant = torch.sqrt(torch.Tensor([2.])).to(x1.device)  # 0,2,3,4,7,8,12
        p_0 = 0.5 * (1 + torch.erf(res[0] / constant))
        p_2 = 0.5 * (1 + torch.erf(res[1] / constant))
        p_3 = 0.5 * (1 + torch.erf(res[2] / constant))
        p_4 = 0.5 * (1 + torch.erf(res[3] / constant))
        p_7 = 0.5 * (1 + torch.erf(res[4] / constant))
        p_8 = 0.5 * (1 + torch.erf(res[5] / constant))
        p_12 = 0.5 * (1 + torch.erf(res[6] / constant))
        return [p_0,F_1[1],p_2,p_3,p_4,F_1[5],F_1[6],p_7,p_8,F_1[9],F_1[10],F_1[11],p_12,F_1[13]]

    


DiffusionExtractor = diffextractor()
DiffusionExtractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(DiffusionExtractor).to(device)
DiffusionExtractor = torch.nn.parallel.DistributedDataParallel(DiffusionExtractor, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=False)
optimizer = torch.optim.Adam(DiffusionExtractor.parameters(),lr=0.0001)


saveidx = 0
def train():
    global saveidx
    DiffusionExtractor.train()
    loss_record=[]
    for imgs,labels in tqdm(TrainDataloader):

        imgs, labels = imgs.to(device), labels.to(device)
        
        outT  = DiffusionExtractor(imgs[:,0,:,:,:,:],imgs[:,1,:,:,:,:])
        
        LossT = 0
        for item in range(14):
            item = int(item)
            out = outT[item]
            if str(item) in list(cls.keys()):
                labels_target = labels[:,item].to(dtype=torch.long)
                loss = CLE(out, labels_target)
                pass
            else:
                labels_target = labels[:,item].unsqueeze(1)
                loss = BCE(out,labels_target)
                pass
            LossT += loss
        optimizer.zero_grad()
        LossT.backward()
        optimizer.step()
        saveidx += 1
        loss_record.append(LossT.item())

        if local_rank == 0 and saveidx % 1000 == 0:
            state = {
                    'state_net':DiffusionExtractor.state_dict(),
                        }

            save_dir = 'ckpt/backbone'
            save_path = os.path.join(save_dir, '{}.pth'.format(saveidx))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)
            torch.save(state, save_path)

    if local_rank == 0 and epoch % 1 == 0:
        print('****** train | epoch:{} | iterion:{} | Loss-a:{:.4f} | ******'.format(epoch,saveidx, np.mean(loss_record)))
        

if __name__ == '__main__':
    for epoch in range(1,12001):
        train_sampler.set_epoch(epoch)
        train()

