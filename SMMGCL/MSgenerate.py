from __future__ import division
from __future__ import print_function

import anndata
import anndata as ad
import pandas as pd
import os
import pdb

import scipy
import sklearn
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import matplotlib

matplotlib.use('Agg')
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

import torch
from typing import Optional
from scipy.sparse import issparse
import warnings

warnings.filterwarnings('ignore')


def tfidf(X):
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def find_column_indices(arr):
    column_indices = []
    for row in arr:
        if np.all(row == 0):
            column_indices.append(5)
        else:
            column_indices.append(np.where(row > 0)[0][0])
    return np.array(column_indices)


def construct_graph_by_feature(feature, k=20, mode="connectivity", metric="correlation",
                               include_self=False):
    adj_f = kneighbors_graph(feature, k, mode=mode, metric=metric,
                             include_self=include_self)

    return adj_f


def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    adj_s = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    adj_s = adj_s.kneighbors_graph(cell_position)
    return adj_s


def construct_adjacency_matrix(adj_wave, prunning_one, prunning_two, common_neighbors):
    if prunning_one:
        # Pruning strategy 1
        original_adj_wave = adj_wave.A
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        np_adj_wave = construct_symmetric_matrix(adj_wave.A)
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()
    adj_hat = construct_adjacency_hat(adj)
    return adj, adj_wave, adj_hat


def construct_adjacency_hat(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def construct_symmetric_matrix(original_matrix):
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
                pdb.set_trace()
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")
        pdb.set_trace()

    return result_matrix


def construct_sparse_float_tensor(np_matrix):
    sp_matrix = sp.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def graph_construction(feature, spatial, dataset, view_no, k, a, prunning_one=True, prunning_two=True,
                       common_neighbors=2):
    adj_s = construct_graph_by_coordinate(spatial, n_neighbors=k)
    adj_f = construct_graph_by_feature(feature, k=k, mode="connectivity", metric="correlation", )
    adj_s = a * adj_s + (1 - a) * adj_f
    adj, adj_wave, adj_hat = construct_adjacency_matrix(adj_s, prunning_one, prunning_two, common_neighbors)

    graph_dict = {
        "adj_hat": adj_hat,
        "adj_wave": adj_wave,
    }
    np.save("../generate_data/" + dataset + "/" + str(view_no) + '_graph_dict.npy', graph_dict)


def clr_normalize_each_cell(adata, inplace=True):
    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def normalize(adata, highly_genes=3000, min_genes=100, min_cells=100):
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return adata


def load_datas(dataset, highly_genes=3000, k=15, a=0.5):
    adata_omics1 = sc.read_h5ad('../data/' + dataset + '/adata_RNA.h5ad')
    adata_omics2 = sc.read_h5ad('../data/' + dataset + '/adata_ADT.h5ad')
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    if dataset == 'Dataset1_Mouse_Spleen1':
        adata_omics1.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata_omics1.obsm['spatial'])).T).T).T
        adata_omics1.obsm['spatial'][:, 1] = -1 * adata_omics1.obsm['spatial'][:, 1]
        adata_omics2.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata_omics2.obsm['spatial'])).T).T).T
        adata_omics2.obsm['spatial'][:, 1] = -1 * adata_omics2.obsm['spatial'][:, 1]
    else:
        adata_omics1.obsm['spatial'] = np.rot90(np.array(adata_omics1.obsm['spatial'])).T
        adata_omics1.obsm['spatial'][:, 0] = -1 * adata_omics1.obsm['spatial'][:, 0]
        adata_omics2.obsm['spatial'] = np.rot90(np.array(adata_omics2.obsm['spatial'])).T
        adata_omics2.obsm['spatial'][:, 0] = -1 * adata_omics2.obsm['spatial'][:, 0]

    def pca(adata, use_reps=None, n_comps=10):
        from sklearn.decomposition import PCA
        from scipy.sparse.csc import csc_matrix
        from scipy.sparse.csr import csr_matrix
        pca = PCA(n_components=n_comps)
        if use_reps is not None:
            feat_pca = pca.fit_transform(adata.obsm[use_reps])
        else:
            if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
                feat_pca = pca.fit_transform(adata.X.toarray())
            else:
                feat_pca = pca.fit_transform(adata.X)

        return feat_pca

    adata_omics1 = normalize(adata_omics1, highly_genes, 0, 10)
    adata_omics2 = clr_normalize_each_cell(adata_omics2)
    sc.pp.scale(adata_omics2)
    feat1 = pca(adata_omics1, n_comps=50)

    if issparse(feat1):
        feat = feat1.toarray()
    else:
        feat = feat1
    adata1 = ad.AnnData(pd.DataFrame(feat, dtype=np.float32))
    adata1.var_names_make_unique()
    adata1.obsm['spatial'] = adata_omics1.obsm['spatial']
    adata1.var_names_make_unique()

    if issparse(adata_omics2.X):
        feat = adata_omics2.X.toarray()
    else:
        feat = adata_omics2.X
    adata2 = ad.AnnData(
        pd.DataFrame(feat, index=adata_omics2.obs.index, columns=np.array(adata_omics2.var.index), dtype=np.float32))
    adata2.var_names_make_unique()
    adata2.obsm['spatial'] = adata_omics2.obsm['spatial']
    adata2.var_names_make_unique()

    graph_construction(adata1.X, adata1.obsm['spatial'], dataset, 0, k, a)
    graph_construction(adata2.X, adata2.obsm['spatial'], dataset, 1, k, a)

    adata1.write("../generate_data/" + dataset + "/" + 'adata_omics1.h5ad')
    adata2.write("../generate_data/" + dataset + "/" + 'adata_omics2.h5ad')


if __name__ == "__main__":
    datasets = ['Dataset1_Mouse_Spleen1']
    # datasets = ['Dataset1_Mouse_Spleen1', 'Dataset2_Mouse_Spleen2']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        generatepath = "../generate_data/" + dataset + "/"
        if not os.path.exists(generatepath):
            os.mkdir(generatepath)
        savepath = '../result/' + dataset + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        load_datas(dataset, highly_genes=3000, k=9, a=0.3)
