import os
import torch

os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
torch.set_num_threads(12)

from oc_funs import *

backbone = 'ire'
eval_noise = 'None'
tag = 'reimplement'


checkpoints_path = opt.backbone_path
net = load_net(path=checkpoints_path)
net.eval()

extract_feature_training(save_dir='oc_dumps/features/train/', net=net, backbone= backbone, tag=tag)
extract_test_features(eval_noise=eval_noise,net=net, save_dir='oc_dumps/features/test/', backbone= backbone, tag=tag)

train_GMM_sklearn(feature_path = 'oc_dumps/features/train/real_train_features_{}_{}.pth'.format(backbone,tag),modelname='gmm_{}_{}.pkl'.format(backbone,tag),path_dir='oc_dumps/oc_models/')
threshold = val_GMM_sklearn(filename="oc_dumps/oc_models/gmm_{}_{}.pkl".format(backbone,tag), valpath = "oc_dumps/features/train/real_train_features_{}_{}.pth".format(backbone,tag))
test_GMM_sklearn(filename="oc_dumps/oc_models/gmm_{}_{}.pkl".format(backbone,tag), feature_dir_path="oc_dumps/features/test/", pred_save='oc_dumps/result/',eval_noise=eval_noise, backbone= backbone, tag=tag)
average_acc,average_map,average_map = GMM_compute_mAP_auc(threshold=threshold,pre_dir_path='oc_dumps/result/', eval_noise=eval_noise,backbone=backbone,save_fig=False,save_log=True, tag=tag)