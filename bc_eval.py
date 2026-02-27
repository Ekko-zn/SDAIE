from config import opt
import os
import torch
torch.set_num_threads(8)


from utils import *
from Classifier import base

import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time



device = 'cuda'

CLE = torch.nn.CrossEntropyLoss()



class combonet(torch.nn.Module):
    def __init__(self):
        super(combonet, self).__init__()
        self.diffusionextractor_pretrained =  base()
        self.diffusionextractor_random =  base()
        self.fc = torch.nn.Linear(528*1,2)
    def forward(self,x):
        f_random,f1,f2,f3,f4,f5 = self.diffusionextractor_random(x)
        out = self.fc(f_random)
        return out



def test():

    CombnNet = combonet()

    checkpoint = opt.bc_ckpt_path
    print(checkpoint)
    state = torch.load(checkpoint,map_location='cpu')['state_net']
    try:
        CombnNet.load_state_dict(state)
    except:
        CombnNet.load_state_dict({k.replace('module.', ''): v for k, v in                 
                            state.items()})


    CombnNet = CombnNet.to(device)
    CombnNet.eval()
    aver_acc = []
    aver_ap = []
    path = opt.test_image_path
    vals = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    for test_id in vals:
        Loss1 = []
        Y_true, Y_pred = [], []
        TestDataset = DatasetforDiffusionGranularityBinary('{}/{}/'.format(opt.test_image_path,test_id),flag='test',patch_type='adaptive')
        TestDataloader = torch.utils.data.DataLoader(TestDataset,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    drop_last = True)
        count = 0
        correct = 0
        for (imgs_diffusion,labels) in tqdm(TestDataloader):
            imgs_diffusion, labels = imgs_diffusion.to(device), labels.to(device).to(dtype=torch.long)
            out1  = CombnNet(imgs_diffusion)
            loss1 = CLE(out1,labels)
            loss = loss1*1 
            out1 = torch.nn.Softmax()(out1)
            y_pred1 = 1-out1[:,0]
            

            Loss1.append(loss1.item())
            Y_true.append(labels.cpu())
            Y_pred.append(y_pred1.cpu())
            pass
        Y_true,Y_pred = torch.stack(Y_true).view(-1,1), torch.stack(Y_pred).view(-1,1)

        acc = accuracy_score(Y_true, Y_pred>0.5)
        ap = average_precision_score(Y_true, Y_pred)

        print('****** test:{} | Loss:{:.2f} | acc:{:.2f} | ap:{:.2f}'.format(test_id, np.mean(Loss1),100*acc, 100*ap))

        aver_acc.append(acc)
        aver_ap.append(ap)
    print('Test | noise :{} | average-acc:{:.2f} | average-ap:{:.2f}'.format(opt.eval_noise, 100*np.mean(aver_acc),  100*np.mean(aver_ap)))

            
with torch.no_grad():
    test()
