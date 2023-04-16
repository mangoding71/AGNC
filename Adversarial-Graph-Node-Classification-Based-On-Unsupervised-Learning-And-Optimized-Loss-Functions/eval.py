import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import networkx as nx
import random

from models import DGI,LogReg
from utils import process
from components.estimator import Calculate_loss
from ThreatModel.attacker import Attacker

import os, argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', default='cora', help='dataset name')   # 'cora', 'citeseer', 'polblogs'
parser.add_argument('--model', default='model.pkl', help='network name')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
dataset = args.dataset
attack_rate = 0.2

hid_units = 512
sparse = True
nonlinearity = 'prelu'

if dataset == 'polblogs':
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data_polblogs(dataset)
else:
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)

if dataset == 'polblogs':
    attack_mode = 'A'
else:
    attack_mode = 'both'

A = adj.copy()
A.setdiag(0)
A.eliminate_zeros()
features, _ = process.preprocess_features(features, dataset=dataset)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
nb_edges = int(adj.sum() / 2)

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:#如果是稀疏
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

model = DGI(ft_size, hid_units, nonlinearity, critic="bilinear", dataset=dataset)
model.eval()

#.cuda()的作用是吧将内存中的数据复制到GPU的显存中去,然后用GPU运行
if torch.cuda.is_available():
    print('CUDA is available!')
    model.cuda()
    features = features.cuda()
    if sparse:
        sparse_adj = sparse_adj.cuda()
        sparse_A = sparse_A.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
else:
    print('CUDA is unavailable!')

original_sparse_adj = sparse_adj.clone()

if attack_mode != 'A':
    features_ori = features.clone()

xent = nn.CrossEntropyLoss()

model.load_state_dict(torch.load(args.model))
print("Load model...")


embeds, _ = model.embed(features, sparse_adj if sparse else adj, sparse, None)
train_embs = embeds[0, idx_train]
val_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)

tot = torch.zeros(1)
if torch.cuda.is_available():
    tot = tot.cuda()

accs = []

for _ in range(50):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    if torch.cuda.is_available():
        log.cuda()


    best_acc = torch.zeros(1)
    if torch.cuda.is_available():
        best_acc = best_acc.cuda()
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
    accs.append(acc * 100)
    tot += acc

accs = torch.stack(accs)
natural_acc_mean = accs.mean().detach().cpu().numpy()
natural_acc_std = accs.std().detach().cpu().numpy()
print(accs.detach().cpu().numpy())
print('Natural accuracy: {} (std: {})'.format(natural_acc_mean, natural_acc_std))


print("== Start Attacking ==")

b_xent = nn.BCEWithLogitsLoss()
atm = Attacker(model, features, nb_nodes, attack_mode=attack_mode,show_attack=False, gpu=torch.cuda.is_available())

if torch.cuda.is_available():
    atm = atm.cuda()
else:
    print('atm is not working on GPU!')

n_flips = int(attack_rate * nb_edges)

acc_list = []
original_sparse_A = sparse_A.clone()
for _ in range(10):
    adv = atm(original_sparse_adj.to_dense(), original_sparse_A.to_dense(), None, n_flips, eps_x=0.1, step_size_x=1e-3,
              b_xent=b_xent, step_size=20, iterations=50, should_normalize=True,
              random_restarts=False, make_adv=True, return_a=False)
    if attack_mode == 'A':
        embeds, _ = model.embed(features, adv, sparse, None)
        loss = Calculate_loss(model, adv, features, nb_nodes, b_xent, 1, sparse)
        original_loss = Calculate_loss(model, original_sparse_adj, features, nb_nodes, b_xent, 1, sparse)
    elif attack_mode == 'X':
        embeds, _ = model.embed(adv, sparse_adj, sparse, None)
        loss = Calculate_loss(model, sparse_adj, adv, nb_nodes, b_xent, 1, sparse)
        original_loss = Calculate_loss(model, original_sparse_adj, features_ori, nb_nodes, b_xent, 1, sparse)
    elif attack_mode == 'both':
        embeds, _ = model.embed(adv[1], adv[0], sparse, None)
        loss = Calculate_loss(model, adv[0], adv[1], nb_nodes, b_xent, 1, sparse)
        original_loss = Calculate_loss(model, original_sparse_adj, features_ori, nb_nodes, b_xent, 1, sparse)

    Training_error = loss - original_loss
    print("Training_error: {}".format(Training_error.detach().cpu().numpy()))

    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1)
    if torch.cuda.is_available():
        tot = tot.cuda()
    else:
        print('tot is not working on the GPU!')
    accs = []

    for _ in range(5):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        if torch.cuda.is_available():
            log.cuda()

        best_acc = torch.zeros(1)
        if torch.cuda.is_available():
            best_acc = best_acc.cuda()
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
        accs.append(acc * 100)
        tot += acc

    accs = torch.stack(accs)
    print("accuracy: {} ".format(accs.mean().detach().cpu().numpy()))
    acc_list += accs.detach().cpu().tolist()

print(acc_list)
print('Adversarial accuracy: {} (std: {})'.format(np.mean(acc_list), np.std(acc_list)))
print('Natural accuracy: {} (std: {})'.format(natural_acc_mean, natural_acc_std))
