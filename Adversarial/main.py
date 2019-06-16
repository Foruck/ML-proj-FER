import h5py
import torch
import numpy as np
import torch.nn as nn
import config
from torch.utils.data import DataLoader
from advertorch.attacks import GradientSignAttack
from advertorch.defenses import GaussianSmoothing2D, AverageSmoothing2D, MedianSmoothing2D
from torchvision import transforms, utils, models
from dataset import FerDataset
from models import ResNet50, LightResNet
from utils import adversarial_train, adversarial_test, adversarial_visualization

def loadData(path):
    raw_data = {}
    with h5py.File(path, 'r') as fp:
        for ds in ['train', 'valid', 'test']:
            raw_data[ds] = {}
            raw_data[ds]['data'] = np.array(fp[ds]['data']).reshape((-1, 48, 48)).astype(np.uint8)
            raw_data[ds]['label'] = np.array(fp[ds]['label'])
    return raw_data

def get_loaders():
    raw_data = loadData(config.pathData)
    transform = {'train': transforms.Compose([
                transforms.ToPILImage(mode='L'),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(config.crop_size),
                transforms.Resize(config.input_size),
                transforms.ToTensor()
            ]),
            'valid':transforms.Compose([
                transforms.ToPILImage(mode='L'),
                transforms.CenterCrop(config.crop_size),
                transforms.Resize(config.input_size),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(mode='L'),
                transforms.CenterCrop(config.crop_size),
                transforms.Resize(config.input_size),
                transforms.ToTensor(),
            ])
        }
    datasets  = {'train': FerDataset(1, raw_data['train'], transform['test']),
            'test': FerDataset(1, raw_data['test'], transform['test']),
            'valid': FerDataset(1, raw_data['valid'], transform['valid'])}
    dataloaders = {"train": DataLoader(datasets['train'], batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers), 
               "valid": DataLoader(datasets['valid'], batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers), 
               "test": DataLoader(datasets['test'], batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers ),
              }
    return dataloaders

def main():
    dataloaders = get_loaders()

    model_res50 = ResNet50(config.input_ch, config.output_ch).to(config.device)
    checkpoint = torch.load(config.ckpt_res50)
    model_res50.load_state_dict(checkpoint['state_dict'])

    model_resli = LightResNet(config.input_ch, config.output_ch).to(config.device)
    checkpoint = torch.load(config.ckpt_light)
    model_resli.load_state_dict(checkpoint['state_dict'])

    FGS_li = GradientSignAttack(model_resli, loss_fn=nn.CrossEntropyLoss(), eps=config.epsilon, clip_min=0.0, clip_max=1.0, targeted=False) # Fast Gradient Sign
    FGS_50 = GradientSignAttack(model_res50, loss_fn=nn.CrossEntropyLoss(), eps=config.epsilon, clip_min=0.0, clip_max=1.0, targeted=False) # Fast Gradient Sign

    adversarial_visualization(model_resli, dataloaders['test'], FGS_li)

    defense = AverageSmoothing2D(1, kernel_size=3).cuda()
    # defense = GaussianSmoothing2D(1, 1).cuda()
    # defense = MedianSmoothing2D().cuda()
    acc1, acc2, acc3 = adversarial_test(model_resli, dataloaders['train'], FGS_50, defense)

    print(acc1, acc2, acc3)

    for i in range(config.num_epoch):
        model_resli, optimizer_li = adversarial_train(model_resli, optimizer_li, dataloaders['train'], i, FGS_li)
        acc1, acc2 = adversarial_test(model_resli, dataloaders['test'], FGS_li)
        print('li-li, w/o %.5f, w %.5f' % (acc1, acc2))
        acc1, acc2 = adversarial_test(model_resli, dataloaders['test'], FGS_50)
        print('li-50, w/o %.5f, w %.5f' % (acc1, acc2))

if __name__ == '__main__':
    main()