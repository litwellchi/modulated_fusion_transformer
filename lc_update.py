import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.optim as optim
import copy
import time

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        id, x, y, z, ans = self.dataset[self.idxs[item]]
        return id, x, y, z, ans


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.args.batch_size, shuffle=True)

    def train(self, epoch, net, criterion, flag, idx, count):
        net.train()
        local_ep = 2
        # Load the optimizer paramters
        optim = torch.optim.Adam(net.parameters(), lr=self.args.lr_base)
        
        for i in range(local_ep):
            batch_loss = []
            time_start = time.time()
            # mini-batch training
            for step, (
                    id,
                    x,
                    y,
                    z,
                    ans,
            ) in enumerate(self.train_loader):
                loss_tmp = 0
                net.zero_grad()

                x = x.cuda()
                y = y.cuda()
                z = z.cuda()
                ans = ans.cuda()

                if flag == 0:
                    pred = net(x, x, x, flag)
                if flag == 1:
                    pred = net(y, y, y, flag)

                loss = criterion(pred, ans)
                loss.backward()

                loss_tmp += loss.cpu().data.numpy()

                print("\r[Epoch %2d][Count %2d][Client %2d][Modality %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, %4d m "
                    "remaining" % (
                        epoch,
                        count,
                        idx,
                        flag,
                        step,
                        int(len(self.train_loader.dataset) / self.args.batch_size),
                        loss_tmp / self.args.batch_size,
                        *[group['lr'] for group in optim.param_groups],
                        ((time.time() - time_start) / (step + 1)) * ((len(self.train_loader.dataset) / self.args.batch_size) - step) / 60,
                    ), end='          ')

                # Gradient norm clipping
                if self.args.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.args.grad_norm_clip
                    )

                optim.step()

        return net.state_dict(), loss_tmp / self.args.batch_size
