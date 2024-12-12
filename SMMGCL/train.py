import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')


def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def cosine_similarity(emb1, emb2):
    mat = torch.matmul(emb1, emb2.T)
    norm1 = torch.norm(emb1, p=2, dim=1).reshape((emb1.shape[0], 1))
    norm2 = torch.norm(emb2, p=2, dim=1).reshape((emb2.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm1, norm2.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def Train(model, feature_list, adj_hat_list, adj_wave_list, args):
    model.cuda()
    for i in range(model.num_views):
        feature_list[i] = feature_list[i].cuda()
        adj_hat_list[i] = adj_hat_list[i].to_dense().cuda()
        adj_wave_list[i] = adj_wave_list[i].cuda()
    if args.optimizer == "RMSprop":
        optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_function = nn.MSELoss()

    model.eval()

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        h, z, adjz, Xz_list, qz, qh = model(feature_list, adj_hat_list)
        optimizer.zero_grad()

        lr_z = []
        lg_z = []
        for v in range(model.num_views):
            lr_z.append(loss_function(feature_list[v], Xz_list[v]))
            lg_z.append(F.binary_cross_entropy(adjz.view(-1), adj_wave_list[v].to_dense().view(-1)))

        loss_rx = sum(lr_z)
        loss_rg = sum(lg_z)

        loss_con = consistency_loss(h, z)

        args.threshold = 0.8
        Q_gloal = torch.mm(qh, qz.t())
        Q_gloal.fill_diagonal_(1)
        pos_mask_gloal = (Q_gloal >= args.threshold).float()
        Q_gloal = Q_gloal * pos_mask_gloal
        Q_gloal = Q_gloal / Q_gloal.sum(1, keepdims=True)

        sim_fusion = torch.sigmoid(cosine_similarity(h, z))
        sim_fusion = sim_fusion / sim_fusion.sum(1, keepdims=True)
        loss_contrast_global = - (torch.log(sim_fusion + 1e-7) * Q_gloal).sum(1)
        loss_cl = loss_contrast_global.mean()

        loss = loss_rx + args.rg_weight * loss_rg + args.con_weight * loss_con + args.cl_weight * loss_cl

        loss.backward()
        optimizer.step(closure=None)
    model.eval()
    h, z, adjz, Xz_list, qz, qh = model(feature_list, adj_hat_list)
    embedding = pd.DataFrame(z.cpu().detach().numpy()).fillna(0).values

    return embedding