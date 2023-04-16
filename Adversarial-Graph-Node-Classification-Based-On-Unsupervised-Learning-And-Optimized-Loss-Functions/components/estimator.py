import torch
import numpy as np

def Calculate_loss(encoder, sparse_adj, features, nb_nodes, b_xent, batch_size=1, sparse=True, encoder_cpu=False):
    idx = np.random.permutation(nb_nodes) #打乱数据集中原有顺序，对应的标签也被打乱
    shuf_fts = features[:, idx, :]

    Tensor1 = torch.ones(batch_size, nb_nodes)#返回一个全为1 的张量
    Tensor2 = torch.zeros(batch_size, nb_nodes)#返回一个全为标量 0 的张量
    Tensor3 = torch.cat((Tensor1, Tensor2), 1)  #将两个张量（tensor）拼接在一起，cat是concatnate的意思

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        Tensor3 = Tensor3.cuda()
    logits = encoder(features, shuf_fts, sparse_adj, sparse, None, None, None)

    loss = b_xent(logits, Tensor3)
    return loss

