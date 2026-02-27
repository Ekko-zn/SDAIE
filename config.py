import argparse
import os
import torch
import random
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--tag', type=str, default='exp')
        parser.add_argument('--test_image_path', type=str, default='/data/AIGI_test/')
        ## backbone_train
        parser.add_argument('--exif_tags_path', type=str, default='/data/SDAIE_gm/exif_tags.json')
        parser.add_argument('--exif_image_path', type=str, default='/data/SDAIE_gm/pkl_256/')
        parser.add_argument('--train_size', type=int, default=900000)
        ## oc train&eval (backbone evaluation)
        parser.add_argument('--oc_realonly_image_path', type=str, default='/data/SDAIE_gm/official_codes_oc/realonly/')
        parser.add_argument('--backbone_path', type=str, default='/data/SDAIE_gm/official_codes_unity/ckpt/backbone/pretrained.pth')
        ## bc_train
        parser.add_argument('--bc_trainset_path', type=str, default='/data/CNNSpotTrainset/')
        parser.add_argument('--alpha', type=float, default=0.05)
        parser.add_argument('--jpg_prob', type=float, default=0.25)
        parser.add_argument('--resize_prob', type=float, default=0.3)
        parser.add_argument('--blur_prob', type=float, default=0.1)
        parser.add_argument('--jpg_low_value', type=int, default=90)
        parser.add_argument('--jpg_high_value', type=int, default=100)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--bc_log_save', type=int, default=1)
        ## bc_eval
        parser.add_argument('--bc_ckpt_path', type=str, default='/data/SDAIE_gm/official_codes_unity/ckpt/bc/pretrained.pth')
        parser.add_argument('--eval_noise', type=str, default='none')
        parser.add_argument('--eval_noise_param', type=float)
        ##
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()



    def parse(self):
        opt = self.gather_options()
        
        return opt


    
opt = BaseOptions().parse()


set_random_seed(opt.seed)
