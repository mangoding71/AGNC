import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from tqdm import tqdm
import networkx as nx
import random
import math, os
from collections import defaultdict

from models import DGI
from models.dgi import task
from utils import process
from ThreatModel.attacker import Attacker
from components.estimator import Calculate_loss

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--dataset', type=str, default='cora', help='dataset')  # 'cora', 'citeseer', 'polblogs'
parser.add_argument('--save-model', type=bool, default=True)
parser.add_argument('--show-task', type=bool, default=False)
parser.add_argument('--show-attack', type=bool, default=False)
args = parser.parse_args()
dataset = args.dataset
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

make_adv = True
attack_rate = 0.4

batch_size = 1

Max_num_iterations = 1
hid_units = 512
sparse = True
if dataset == 'polblogs':
    attack_mode = 'A'
else:
    attack_mode = 'both'
nonlinearity = 'prelu'

if dataset == 'polblogs':
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data_polblogs(dataset)
else:
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
nb_edges = int(adj.sum() / 2)
n_flips = int(nb_edges * attack_rate)

A = adj.copy()
features, _ = process.preprocess_features(features, dataset=dataset)

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sparse_adj = process.Convert_To_Torch_Tensor(adj)
    sparse_A = process.Convert_To_Torch_Tensor(A)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


if torch.cuda.is_available():
    print('CUDA is available!')
    if sparse:
        sparse_adj = sparse_adj.cuda()
        sparse_A = sparse_A.cuda()
    else:
        adj = adj.cuda()
        A = A.cuda()

    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
else:
    print('CUDA is unavailable!')

sparse_adj = sparse_adj.to_dense()
original_sparse_adj = sparse_adj.clone()
original_features = features.clone()
sparse_A = sparse_A.to_dense()


encoder = DGI(ft_size, hid_units, nonlinearity, critic="bilinear")

atm = Attacker(encoder, features, nb_nodes, attack_mode=attack_mode,show_attack=args.show_attack, gpu=torch.cuda.is_available())

#lr (float, 可选) – 学习率（默认：1e-3）,weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
optimiser = torch.optim.Adam(encoder.parameters(), lr=0.001, weight_decay=0)

if torch.cuda.is_available():
    encoder.cuda()
    atm.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()

Current_num_iterations = 0
Best_loss_adjustment = 1e9
best_t = 0

step_size_init = 20
attack_iters = 10
stepsize_x = 1e-5

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)
nb_epochs = 10000
for epoch in range(nb_epochs):
    encoder.train()
    optimiser.zero_grad()

    if make_adv:
        step_size = step_size_init
        step_size_x = stepsize_x
        adv = atm(sparse_adj, sparse_A, None, n_flips, b_xent=b_xent, step_size=step_size,
                  eps_x=0.1, step_size_x=step_size_x,
                  iterations=attack_iters, should_normalize=True, random_restarts=False, make_adv=True)
        if attack_mode == 'A':
            sparse_adj = adv
        elif attack_mode == 'X':
            features = adv
        elif attack_mode == 'both':
            sparse_adj = adv[0]
            features = adv[1]

    loss = Calculate_loss(encoder, sparse_adj, features, nb_nodes, b_xent, batch_size, sparse)
    if True:#arge.hinge
        original_loss = Calculate_loss(encoder, original_sparse_adj, original_features, nb_nodes, b_xent, batch_size, sparse)
        Training_error = loss - original_loss
        print("Training_error: {}; Training_error-tau: {}; MI-nature: {}; MI-worst: {}".format(Training_error.detach().cpu().numpy(),
                                                                       (Training_error - 0.01).detach().cpu().numpy(),
                                                                       original_loss.detach().cpu().numpy(),
                                                                       loss.detach().cpu().numpy()))
        if Training_error - 0.01 < 0:
            loss = original_loss

    if args.show_task and epoch%5==0:
        adv = atm(original_sparse_adj, sparse_A, None, n_flips, b_xent=b_xent, step_size=20,
                  eps_x=0.1, step_size_x=1e-3,
                  iterations=50, should_normalize=True, random_restarts=False, make_adv=True)
        if attack_mode == 'A':
            embeds, _ = encoder.embed(original_features, adv, sparse, None)
        elif attack_mode == 'X':
            embeds, _ = encoder.embed(adv, original_sparse_adj, sparse, None)
        elif attack_mode == 'both':
            embeds, _ = encoder.embed(adv[1], adv[0], sparse, None)
        acc_adv = task(embeds,idx_train, idx_val, idx_test,hid_units, nb_classes,train_lbls,xent,test_lbls)

        embeds, _ = encoder.embed(original_features, original_sparse_adj, sparse, None)
        acc_nat = task(embeds,idx_train, idx_val, idx_test,hid_units, nb_classes,train_lbls,xent,test_lbls)

        print('Epoch:{} Step_size: {:.4f} Loss:{:.4f} Natural_Acc:{:.4f} Adv_Acc:{:.4f}'.format(
            epoch, step_size, loss.detach().cpu().numpy(), acc_nat, acc_adv))
    else:
        print('Epoch:{} Step_size: {:.4f} Loss:{:.4f}'.format(epoch, step_size, loss.detach().cpu().numpy()))

    if loss < Best_loss_adjustment:#当模型计算的损失小于目前最小的损失，就保存当前的模型
        Best_loss_adjustment = loss
        best_t = epoch
        Current_num_iterations = 0 #迭代次数归零，也就是每次获得的新模型都要迭代“Max_num_iterations”次
        if args.save_model:
            torch.save(encoder.state_dict(), 'model.pkl')
    else:
        Current_num_iterations += 1
        print(Current_num_iterations)
    if Current_num_iterations == Max_num_iterations:
        print('模型训练完毕!')
        break

    loss.backward()
    optimiser.step()
