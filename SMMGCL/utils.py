import os
import torch
from typing import Optional, List
from collections import OrderedDict
from cytoolz.itertoolz import sliding_window
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class GraphConvolution(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.initial_parameter(bias)

    def initial_parameter(self, bias: bool) -> None:
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        initial_weight = torch.rand(self.input_dim, self.output_dim) * 2 * init_range - init_range
        self.weight = nn.Parameter(initial_weight)
        if bias:
            initial_bias = torch.rand(self.output_dim) * 2 * init_range - init_range
            self.bias = nn.Parameter(initial_bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, fea: torch.Tensor, adj: torch.sparse) -> torch.Tensor:
        hidden = torch.mm(fea, self.weight)
        output = torch.spmm(adj, hidden)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


def build_layer_units(layer_type: str, dims: List[int], act_func: Optional[nn.Module]) -> nn.Module:
    layer_list = []
    for input_dim, output_dim in sliding_window(2, dims[:-1]):
        layer_list.append(single_unit(layer_type, input_dim, output_dim, act_func))
    layer_list.append(single_unit(layer_type, dims[-2], dims[-1], None))
    return nn.Sequential(*layer_list)


def single_unit(layer_type: str, input_dim: int, output_dim: int, act_func: Optional[nn.Module]):
    unit = []
    if layer_type == 'linear':
        unit.append(('linear', nn.Linear(input_dim, output_dim)))
    elif layer_type == 'gcn':
        unit.append(('gcn', GraphConvolution(input_dim, output_dim)))
    else:
        print("Please input correct layer type!")
        exit()

    if act_func is not None:
        unit.append(('act', act_func))

    return nn.Sequential(OrderedDict(unit))


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def all_fg_consstruct(features, adj_new, true_viewnum):
    view_graph = torch.eye(features[0].shape[0]).cuda()
    view_graph = torch.repeat_interleave(view_graph, repeats=true_viewnum, dim=1)
    view_graph = torch.repeat_interleave(view_graph, repeats=true_viewnum, dim=0)

    for i in range(true_viewnum):
        if i == 0:
            adj_graph = adj_new[0]
        else:
            adj_graph = adjConcat(adj_graph, adj_new[i])
    adj_new = adj_graph + view_graph
    adj_new = adj_new - torch.eye(features[0].shape[0] * true_viewnum).cuda()
    return adj_new


def adjConcat(a, b):
    lena = a.shape[0]
    lenb = b.shape[0]
    p = torch.zeros((lenb, lena)).cuda()
    q = torch.zeros((lena, lenb)).cuda()
    left = torch.vstack((a.to_dense(), p))
    right = torch.vstack((q, b.to_dense()))
    result = torch.hstack((left, right))
    return result


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='embedding', random_seed=2020):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    embedding = adata.obsm[used_obsm]
    embedding = rpy2.robjects.numpy2ri.numpy2rpy(embedding)
    res = rmclust(embedding, num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata
