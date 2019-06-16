import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def adversarial_train(model, optimizer, loader, epoch, perm=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(loader):
        x = x.float().cuda()
        y = y.long().cuda()
        if perm is not None and np.random.rand() > 0.5:
            x = perm(x, y)
        score = model(x)
        loss = loss_fn(score, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 100 == 0):
            print("Epoch {}, iter {}, loss {}".format(epoch, i, loss.detach()))
    return model, optimizer

def adversarial_visualization(model, loader, perm=None):
    names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Normal']
    model.eval()
    perm_x, ori_x = None, None
    perm_y, ori_y = None, None
    perm_conf, ori_conf = None, None
    for i, (x, y) in enumerate(loader):
        x = x.float().cuda()
        y = y.long().cuda()
        score = model(x)
        res = torch.argmax(score, dim=1)
        conf = F.softmax(score)[range(32), list(res)]
        ori_x = x.detach().cpu().numpy()
        ori_y = res.detach().cpu().numpy()
        ori_conf = conf.detach().cpu().numpy()
        if perm is not None:
            x = perm(x, y)
            score = model(x)
            res = torch.argmax(score, dim=1)
            conf = F.softmax(score)[range(32), list(res)]
            perm_x = x.detach().cpu().numpy()
            perm_y = res.detach().cpu().numpy()
            perm_conf = conf.detach().cpu().numpy()
        break
    for i in range(perm_x.shape[0]):
        if ori_y[i] == perm_y[i]:
            continue
        plt.subplot(121)
        plt.imshow(ori_x[i, 0, :, :], cmap='gray')
        plt.title("%s with conf %.4f" % (names[ori_y[i]], ori_conf[i]))
        plt.subplot(122)
        plt.imshow(perm_x[i, 0, :, :], cmap='gray')
        plt.title("%s with conf %.4f" % (names[perm_y[i]], perm_conf[i]))
        plt.savefig('vis.jpg')
        break
def adversarial_test(model, loader, perm=None, defense=None):
    model.eval()
    cnt = 0.0
    hit_p = 0.0
    hit_d = 0.0
    hit = 0.0
    for i, (x, y) in enumerate(loader):
        cnt += x.shape[0]
        x = x.float().cuda()
        y = y.long().cuda()
        score = model(x)
        res = torch.argmax(score, dim=1)
        hit += int(torch.sum(res == y))
        if perm is not None:
            x_p = perm(x, y)
            score = model(x_p)
            res = torch.argmax(score, dim=1)
            hit_p += int(torch.sum(res == y))
            if defense is not None:
                x_d = defense(x_p)
                score = model(x_d)
                res = torch.argmax(score, dim=1)
                hit_d += int(torch.sum(res == y))
    if perm is not None:
        if defense is not None:
            return hit / float(cnt), hit_p / float(cnt), hit_d / float(cnt)
        return hit / float(cnt), hit_p / float(cnt)
    else:
        return hit / float(cnt)