import torch
import numpy as np
from torch.utils.data import Dataset

class FerDataset(Dataset):
    def __init__(self, input_ch, data_label, transform=None):
        self.data = data_label['data']
        self.label= torch.LongTensor(data_label['label'])
        self.transform = transform
        self.input_ch = input_ch

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        dt = self.data[idx,:]
        dt = np.stack([dt]*self.input_ch, axis=2)
        dt = self.transform(dt)
        return dt, self.label[idx]