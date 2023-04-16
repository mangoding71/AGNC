import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np

from .logreg import LogReg
from components import GCN, AvgReadout, Discriminator, Discriminator2, Discriminator_jsd, MeanAggregator, MeanAggregator_ml, Encoder
from ThreatModel.attacker import Attacker

def task(embeds,idx_train, idx_val, idx_test,hid_units, nb_classes,train_lbls,xent,test_lbls):
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    if torch.cuda.is_available():
        log.cuda()

    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    return acc.detach().cpu().numpy()

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, critic="bilinear", dataset=None, attack_model=True):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout() #读出函数，其实这里就是所有节点表示的均值

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h, critic=critic, dataset=dataset, attack_model=attack_model) #判别器，定义为一个双线性函数bilinear

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)   # (1, 2708, 512)

        c = self.read(h_1, msk)   # (1, 512)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)   # (1, 5416)

        return ret
    def embed(self, seq, adj, sparse, msk, grad=False):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        if grad:
            return h_1, c
        else:
            return h_1.detach(), c.detach() #将tensor从计算图中分离出来，不参与反向传播



