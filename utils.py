from config import opt
import torch
import torchvision
from PIL import Image
import os
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from scipy.ndimage.filters import gaussian_filter
import cv2
from io import BytesIO
from random import choice
from itertools import islice
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
from copy import deepcopy
ImageFile.LOAD_TRUNCATED_IMAGES = True


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

def pil_jpg_eval(img, compress_val):
    out = BytesIO()
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    img = Image.fromarray(img)
    out.close()
    return img



jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)




def custom_augment(img):

    if random.random() < opt.resize_prob:
        size = random.randint(64,128)
        img = torchvision.transforms.Resize((size,size))(img)
    
    img = np.array(img)
    if random.random() < opt.blur_prob:
        sig = sample_continuous([0.0,1.0])
        gaussian_blur(img, sig)

    if  random.random() < opt.jpg_prob:
        method = sample_discrete(['cv2','pil'])
        qual = sample_discrete([i for i in range(opt.jpg_low_value,opt.jpg_high_value)])
        img = jpeg_from_key(img, qual, method)


    return Image.fromarray(img)





class DatasetforDiffusionGranularityForEXIF():
    def __init__(self,  flag ,num_block=16) -> None:
        with open(opt.exif_tags_path) as f:
            self.dict_train = json.load(f)
        # with open('./info_test_all_tags.json') as f:
        #     self.dict_test = json.load(f)
        self.totensor = torchvision.transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.patchsize = 64
        self.randomcrop = torchvision.transforms.RandomCrop(self.patchsize)
        self.flag = flag
        self.type = type
        self.imgfiles = self._processpath()
        self.num_block = num_block


        self.names = [t for t in self.imgfiles.keys()][0:opt.train_size]

            
    def _processpath(self):
        
        if self.flag == 'train':
            return self.dict_train
        elif self.flag == 'test':
            return self.dict_test

    def _adaptivepatch(self,imgpath):
        # img = Image.open(imgpath).convert('RGB')
        img = torch.load(imgpath).to(dtype=torch.float) / 255

        c, w,h = img.shape
        if min(w,h) < self.patchsize:
            img = torchvision.transforms.Resize((self.patchsize,self.patchsize))(img)
        else:
            w = w // self.patchsize * self.patchsize
            h = h // self.patchsize * self.patchsize
            img = torchvision.transforms.CenterCrop((w,h))(img)
        img = self.totensor(img)
        patches = img.unfold(1, self.patchsize, self.patchsize).unfold(2, self.patchsize, self.patchsize)
        num_patches = patches.shape[1] * patches.shape[2]
        patches = patches.contiguous().view(3, num_patches, self.patchsize, self.patchsize)
        patches = patches.permute(1,0,2,3)
        return patches
    


    def _fixpatch(self,imgpath):

        img = torch.load(imgpath).to(dtype=torch.float) / 255
        img = self.totensor(img)

        img_crops = []
        for i in range(16):
            cropped_img = self.randomcrop(img)
            img_crops.append(cropped_img)


        img_crops = torch.stack(img_crops)
        img_crops = img_crops.to(dtype=torch.float)
        
        return img_crops
    
    
    def __getitem__(self,index):

        success = False
        while not success:
            try:
                path1 = opt.exif_image_path + self.names[index].replace('jpg','pkl')   
                patches_1 = self._fixpatch(path1)
                success = True
            except:
                index = random.randint(0, len(self.imgfiles))
        success = False    
        while not success:
            try:
                index2 = random.randint(0, len(self.imgfiles))
                path2 = opt.exif_image_path + self.names[index2].replace('jpg','pkl')   
                patches_2 = self._fixpatch(path2)
                success = True
            except:
                index2 = random.randint(0, len(self.imgfiles))


  

        exif_1 = self.imgfiles[self.names[index]]

        exif_2 = self.imgfiles[self.names[index2]]


        img_crops = torch.cat((patches_1.unsqueeze(0),patches_2.unsqueeze(0)),0)

        label = []
        exif_1_values = list(exif_1.values())[3:]  
        exif_2_values = list(exif_2.values())[3:]  
        for x, y in zip(exif_1_values, exif_2_values):
            if x > y or x==y:
                label.append(1)
            elif x < y:
                label.append(0)
            # else:
            #     label.append(0.5)

        label[1],label[5],label[6],label[9],label[10],label[11],label[13] = exif_1['Make'], exif_1['Model'], exif_1['Flash'], exif_1['Metering Mode'], exif_1['Exposure Mode'], exif_1['White Balance Mode'], exif_1['Scene Capture Type']


        label = torch.tensor(label).to(dtype=torch.float)

        return img_crops, label

    def __len__(self):


        return len(self.names)




class DatasetforDiffusionGranularityOC():
    def __init__(self, root, label = '0_real', labelbasedselection=True, max_num = 2000, mode = 'ire', eval_noise = None) -> None:
        self.root = root
        self.totensor = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.mode = mode
        self.max_num = max_num
        self.patchsize = 64
        self.randomcrop = torchvision.transforms.RandomCrop(self.patchsize)
        self.label = label
        self.labelbasedselection = labelbasedselection
        self.paths = self._preprocess() 
        self.eval_noise = eval_noise
        

    def _preprocess(self):
        paths = []
        for root,dirs,files in os.walk(self.root):
            if len(files) != 0:
                for file in files:
                    if self.labelbasedselection == True:
                        if self.label in root:
                            paths.append(root+'/'+file)
                    else:
                        paths.append(root+'/'+file)
        random.seed(42)
        random.shuffle(paths)

        return paths[0:self.max_num]
    
    def _adaptivepatch(self,img):
        # img = torchvision.transforms.Resize((256,256))(img)
        w,h = img.size
        if min(w,h) < self.patchsize:
            img = torchvision.transforms.Resize((self.patchsize,self.patchsize))(img)
        else:
            w = w // self.patchsize * self.patchsize
            h = h // self.patchsize * self.patchsize
            img = torchvision.transforms.CenterCrop((w,h))(img)
        img = self.totensor(img)
        patches = img.unfold(1, self.patchsize, self.patchsize).unfold(2, self.patchsize, self.patchsize)
        num_patches = patches.shape[1] * patches.shape[2]
        patches = patches.contiguous().view(3, num_patches, self.patchsize, self.patchsize)
        patches = patches.permute(1,0,2,3)
        if patches.shape[0] > 20000:
            print(patches.shape[0])
            patches = patches[0:20000,:,:,:]
            
        return patches
    
    
    def __getitem__(self,index):

        path = self.paths[index]
        if '0_real' in path:
            label = 0
        else:
            label = 1
        
        img = Image.open(path).convert('RGB')
        if self.eval_noise == 'train':
            # img = custom_augment(img)
            img = img
        elif self.eval_noise == 'jpg':
            img = pil_jpg_eval(img,int(95))
        elif self.eval_noise == 'resize':
            height,width=img.height,img.width
            img = torchvision.transforms.Resize((int(height*0.5),int(width*0.5)))(img)
        elif self.eval_noise == 'blur':
            img = np.array(img)
            gaussian_blur(img, 1.0)
            img = Image.fromarray(img)
        elif self.eval_noise == 'None':
            img = img
        
        if 'ire' in self.mode:
            img_crops = self._adaptivepatch(img)



        return img_crops, label

    def __len__(self):
        return len(self.paths)

class DatasetforDiffusionGranularityBinary():
    def __init__(self, root, flag = 'train', patch_type = 'fix') -> None:
        self.root = root
        self.totensor = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.totensor_vit = torchvision.transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.paths = self._preprocess() 
        # random.shuffle(self.paths)
        # self.paths = self.paths[:2000]
        self.flag = flag
        self.patch_type = patch_type
        self.patchsize = 64
        self.randomcrop = torchvision.transforms.RandomCrop(self.patchsize)

    def _preprocess(self):
        paths = []
        for root,dirs,files in os.walk(self.root):
            if len(files) != 0:
                for file in files:
                    paths.append(root+'/'+file)
        return paths
        
    def _fixpatch(self,img):
        # img = Image.open(imgpath).convert('RGB')
        if self.flag == 'train':
            img = custom_augment(img)
        elif self.flag =='test':
            if opt.eval_noise == 'jpg':
                img = pil_jpg_eval(img,int(opt.eval_noise_param))
            elif opt.eval_noise == 'resize':
                height,width=img.height,img.width
                img = torchvision.transforms.Resize((int(height*opt.eval_noise_param),int(width*opt.eval_noise_param)))(img)
            elif opt.eval_noise == 'blur':
                img = np.array(img)
                gaussian_blur(img, opt.eval_noise_param)
                img = Image.fromarray(img)
        minsize = min(img.size)
        if minsize < self.patchsize:
            img = torchvision.transforms.Resize((self.patchsize,self.patchsize))(img)
        img = self.totensor(img)
        img_crops = []
        for i in range(16):
            cropped_img = self.randomcrop(img)
            img_crops.append(cropped_img)
        img_crops = torch.stack(img_crops)
        return img_crops
    
    def _adaptivepatch(self,img):
        # img = Image.open(imgpath).convert('RGB')
        if self.flag == 'train':
            img = custom_augment(img)
        elif self.flag =='test':
            if opt.eval_noise == 'jpg':
                img = pil_jpg_eval(img,int(opt.eval_noise_param))
            elif opt.eval_noise == 'resize':
                height,width=img.height,img.width
                img = torchvision.transforms.Resize((int(height*opt.eval_noise_param),int(width*opt.eval_noise_param)))(img)
            elif opt.eval_noise == 'blur':
                img = np.array(img)
                gaussian_blur(img, opt.eval_noise_param)
                img = Image.fromarray(img)

        
        # w,h = img.size
        # if w != 256 or h != 256:
        #     img = torchvision.transforms.Resize((256,256))(img)

        w,h = img.size
        if min(w,h) < self.patchsize:
            img = torchvision.transforms.Resize((self.patchsize,self.patchsize))(img)
        else:
            w = w // self.patchsize * self.patchsize
            h = h // self.patchsize * self.patchsize
            img = torchvision.transforms.CenterCrop((w,h))(img)
        img = self.totensor(img)
        patches = img.unfold(1, self.patchsize, self.patchsize).unfold(2, self.patchsize, self.patchsize)
        num_patches = patches.shape[1] * patches.shape[2]
        patches = patches.contiguous().view(3, num_patches, self.patchsize, self.patchsize)
        patches = patches.permute(1,0,2,3)
        if patches.shape[0] > 40000:
            patches = patches[0:40000,:,:,:]
        return patches
    
    def __getitem__(self,index):

        path = self.paths[index]
        if '0_real' in path:
            label = 0
        else:
            label = 1
        img = Image.open(path).convert('RGB')
        # size = min(img.size) 
    
        if self.patch_type == 'fix':
            img_crops = self._fixpatch(img)
            return img_crops, label
        elif self.patch_type == 'adaptive':
            img_crops = self._adaptivepatch(img)
            return img_crops, label
    def __len__(self):
        return len(self.paths)



def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    return message


