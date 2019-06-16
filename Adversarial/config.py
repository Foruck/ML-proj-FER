import torch
import numpy as np

# config

learning_rate = 1e-4
crop_size  = 44
input_size = 44 
input_ch   = 1
batch_size = 32
output_ch  = 7
num_workers = 8
epsilon    = 0.01
num_epoch  = 10
pathData   = 'drive/fer/data.h5'
ckpt_light = 'drive/fer/resnet50/best.pth.tar'
ckpt_res50 = 'drive/fer/lightres/best.pth.tar'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')