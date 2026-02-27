import os
import pdb

import torch
import torch.utils
from Classifier import base

import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import *
from sklearn import svm
import pickle, time
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.mixture import GaussianMixture
from matplotlib import font_manager

font = font_manager.FontProperties(family='Times New Roman', size=22)

dataset_ls = [
    'progan','stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
        'stylegan2', 'whichfaceisreal',
        'ADM','Glide','Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong','DALLE2','coco_sdxl_nw'
]


description_id = {'progan':'ProGAN','stylegan':'StyleGAN','biggan':'BigGAN', 'cyclegan':'CycleGAN', 'stargan':'StarGAN', 'gaugan':'GauGAN', 'stylegan2':'StyleGAN2',
    'whichfaceisreal':'WFIR', 'ADM':'ADM',
    'Glide':'Glide','Midjourney':'Midjourney','stable_diffusion_v_1_4':'SDv1.4','stable_diffusion_v_1_5':'SDv1.5','VQDM':'VQDM','wukong':'wukong','DALLE2':'DALLE2','coco_sdxl_nw':'SDXL'}




def load_net(path):
    DiffusionExtractor = base()
    checkpoint = path
    netparameters = {}
    for k , v in torch.load(checkpoint)['state_net'].items():
        if 'backbone' in k:
            n = k.replace('module.backbone.', '')
            netparameters[n] = v
    DiffusionExtractor.load_state_dict(netparameters)
    DiffusionExtractor = DiffusionExtractor.cuda()
    return DiffusionExtractor


def save_features(process_features, save_name, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    savepath = os.path.join(save_dir, save_name + '.pth')
    torch.save(process_features.cpu(), savepath)

def extract_feature_training(save_dir, net, backbone, tag,num=5000):
    TestDataset = DatasetforDiffusionGranularityOC(root=opt.oc_realonly_image_path+'/imagenet/',labelbasedselection=False, max_num = num, mode = backbone,eval_noise='train')
    TestDataset2 = DatasetforDiffusionGranularityOC(root=opt.oc_realonly_image_path+'/lsun/', labelbasedselection=False, max_num = num, mode = backbone,eval_noise='train')
    

    Dataset = torch.utils.data.ConcatDataset([TestDataset,TestDataset2])
    TestDataloader = torch.utils.data.DataLoader(Dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 drop_last=True)
    fea_all = []
    for data in tqdm(TestDataloader):
        with torch.no_grad():
            f = net(data[0].cuda())[0]
            fea_all.append(f)

    fea_all = torch.cat(fea_all, dim=0)
    save_name = 'real_train_features_{}_{}'.format(backbone,tag)
    save_features(fea_all, save_name, save_dir)


def extract_test_features(eval_noise, net, save_dir, backbone ,tag):



    for dataset_item in dataset_ls:
        TestDataset_real = DatasetforDiffusionGranularityOC(root='{}/{}'.format(opt.test_image_path,dataset_item),label='0_real',labelbasedselection=True,max_num = 100000, mode = backbone, eval_noise=eval_noise)
        TestDataloader_real = torch.utils.data.DataLoader(TestDataset_real,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     num_workers=0,
                                                     drop_last=True)

        TestDataset_fake = DatasetforDiffusionGranularityOC(root='{}/{}'.format(opt.test_image_path,dataset_item),label='1_fake',labelbasedselection=True,max_num = 100000, mode = backbone, eval_noise=eval_noise)
        TestDataloader_fake = torch.utils.data.DataLoader(TestDataset_fake,
                                                          batch_size=1,
                                                          shuffle=True,
                                                          num_workers=0,
                                                          drop_last=True)

        print(f'length of {dataset_item} - real: {len(TestDataset_real)} - fake: {len(TestDataset_fake)}')

        fea_all = []
        for data in tqdm(TestDataloader_real):
            with torch.no_grad():

                f = net(data[0].cuda())[0]
                fea_all.append(f)

        fea_all = torch.cat(fea_all, dim=0)
        save_name = dataset_item + '_real_{}_{}_{}'.format( backbone, eval_noise,tag )
        save_features(fea_all, save_name, save_dir)

        fea_all = []
        for data in tqdm(TestDataloader_fake):
            with torch.no_grad():
                f = net(data[0].cuda())[0]
                fea_all.append(f)
        fea_all = torch.cat(fea_all, dim=0)
        save_name = dataset_item + '_fake_{}_{}_{}'.format( backbone, eval_noise,tag )
        save_features(fea_all, save_name, save_dir)

        print(f'{dataset_item} features saved...')




def train_GMM_sklearn(feature_path,modelname,path_dir):
    gmm = GaussianMixture(n_components=5 ,init_params='k-means++', random_state=42)


    features = torch.load(feature_path)

    print(features.shape)

    start_time = time.time()
    try:
        gmm.fit(features.detach().cpu().numpy())
    except:
        gmm.fit(features.numpy())
    end_time = time.time()
    runtime = end_time - start_time

    # 转换为小时、分钟和秒
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = int(runtime % 60)

    # 打印运行时间
    print("time: {}hr {}min {}s".format(hours, minutes, seconds))


    os.makedirs(path_dir, exist_ok=True)
    with open(path_dir + modelname, 'wb') as file:
        pickle.dump(gmm, file)





def val_GMM_sklearn(filename,valpath):
    with open(filename, 'rb') as file:
        gmm = pickle.load(file)



    features_real = torch.load(valpath)
    

    print(f'real: {len(features_real)}')

    real_logp = []
    log_likelihoods_real = gmm.score_samples(features_real.cpu())
    real_logp.extend(log_likelihoods_real.tolist())
    real_logp = sorted(real_logp)
    threshold_index = int(len(real_logp) * 0.02) 
    threshold = real_logp[threshold_index]

    print('threshold:{}'.format(threshold))
    return threshold



def test_GMM_sklearn(filename="./checkpoints/gmm.pkl",
                       feature_dir_path="./features/test/",
                       pred_save='./result/',eval_noise='None', backbone= 'ire' , num=None, tag=None):
    with open(filename, 'rb') as file:
        gmm = pickle.load(file)


    for dataset_item in dataset_ls:
        # load the feature data
        feature_path_real = feature_dir_path+dataset_item+'_real_{}_{}_{}.pth'.format(backbone,eval_noise,tag)
        features_real = torch.load(feature_path_real)
        feature_path_fake = feature_dir_path + dataset_item + '_fake_{}_{}_{}.pth'.format(backbone,eval_noise,tag)
        features_fake = torch.load(feature_path_fake)

        print(f'length of {dataset_item} - real: {len(features_real)} - fake: {len(features_fake)}')

        real_logp = []
        log_likelihoods_real = gmm.score_samples(features_real.cpu())
        real_logp.extend(log_likelihoods_real.tolist())

        fake_logp = []
        log_likelihoods = gmm.score_samples(features_fake.cpu())
        fake_logp.extend(log_likelihoods.tolist())

        log_likelihood = {}
        log_likelihood['real'] = real_logp
        log_likelihood[dataset_item] = fake_logp

        # save likelihood
        savename = 'likelihood_'+dataset_item + '_{}_{}_{}'.format(backbone,eval_noise,tag)
        os.makedirs(pred_save, exist_ok=True)
        filename = pred_save + savename+'.pickle'
        with open(filename, 'wb') as file:
            pickle.dump(log_likelihood , file)

        print(f"数据已保存到文件: {filename}")




def plot_distribution(list1, list2, name, backbone, map_score):
    """
    Plots the distribution of elements in two lists.

    Parameters:
    - list1: List of float numbers.
    - list2: List of float numbers.
    """

    
    threshold = -4000

    list1 = [x if x >= threshold else threshold for x in list1]
    list2 = [x if x >= threshold else threshold for x in list2]
    plt.figure(figsize=(6, 6))

    # Plotting the distributions
    plt.hist(list1, bins=50, alpha=0.7, label='List 1', color=(217/255, 129/255, 147/255))
    plt.hist(list2, bins=50, alpha=0.7, label='List 2', color=(157/255, 166/255, 223/255))

    plt.xlim(threshold, 1500)

    # Set font properties for x and y ticks
    font_properties = font_manager.FontProperties(family='Times New Roman', size=18)

    # Set x and y ticks with specific intervals
    plt.xticks(fontproperties=font_properties)  # Adjust interval as needed
    plt.yticks(fontproperties=font_properties)  # Adjust interval as needed

    # Adding title and labels (uncomment if needed)
    # plt.title('Distribution of Elements in Two Lists', fontproperties=font_properties)
    # plt.xlabel('mAP: {:.2f}'.format(map_score), fontproperties=font_properties)
    # plt.ylabel('Frequency', fontproperties=font_properties)

    # Adding legend (uncomment if needed)
    # plt.legend(loc='upper right', prop=font_properties)

    # Showing the plot
    plt.tight_layout()
    plt.savefig('./fig/distribution_bk/{}.pdf'.format(name))

def GMM_compute_mAP_auc(threshold,pre_dir_path='./result/',eval_noise='None',backbone='ire',save_fig=False,save_log=True,epoch=None,num=None,tag=None):
    description_id = [
        'ProGAN', 'StyleGAN', 'BigGAN', 'CycleGAN', 'StarGAN', 
        'GauGAN', 'StyleGAN2', 'WFIR', 'ADM', 'Glide', 
        'Midjourney', 'SDv1.4', 'SDv1.5', 'VQDM', 'wukong', 
        'DALLE2', 'SDXL'
    ]
    
    average_map = []
    average_acc = []
    description = []
    for idx, dataset_item in enumerate(dataset_ls) :
        logp = []
        label = []
        logp_dict_path = pre_dir_path +"likelihood_"+dataset_item+"_{}_{}_{}.pickle".format(backbone, eval_noise,tag)
        with open(logp_dict_path, 'rb') as file:
            likelihood_dict = pickle.load(file)

        # with open('./fig/likelihood/{}'.format(description_id[idx]), 'wb') as file:
        #     pickle.dump(likelihood_dict , file)

        real_logp = likelihood_dict['real']
        fake_logp = likelihood_dict[dataset_item]

        real_label = [1] * len(real_logp)
        fake_label = [0] * len(fake_logp)
        logp.extend(real_logp)
        logp.extend(fake_logp)
        label.extend(real_label)
        label.extend(fake_label)

        label, logp = label, logp
        pred = []
        for item in logp:
            if item > threshold:
                pred.append(1)
            else:
                pred.append(0)

        acc = accuracy_score(label, pred)* 100
        map_score = average_precision_score(label, logp) * 100
        if save_fig == True:
            plot_distribution(real_logp,fake_logp,description_id[idx],backbone,map_score)
        print(f"{dataset_item} Acc: {acc:.2f} mAP: {map_score:.2f}")
        description.append(f"{dataset_item} | Acc: {acc:.2f} | mAP: {map_score:.2f}")
        average_acc.append(acc)
        average_map.append(map_score)
    description.append('{} | average: Acc:{:.2f} mAP:{:.2f}'.format(eval_noise,np.mean(average_acc), np.mean(average_map)))
    description.append('method:{} noise:{}'.format(backbone, eval_noise))
    if save_log:
        log_dir = "./log"
        file_path = os.path.join(log_dir, "{}_{}_{}.txt".format(backbone, eval_noise, tag))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            for item in description:
                file.write(item + "\n")
    print('{} | average: Acc:{:.2f} mAP:{:.2f} \n'.format(eval_noise,np.mean(average_acc), np.mean(average_map)))
    return np.mean(average_acc),np.mean(average_map), description

if __name__=='__main__':
    # extract_test_features()
    pass
