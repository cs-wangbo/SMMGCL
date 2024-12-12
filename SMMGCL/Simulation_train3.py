import os
import argparse
import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')
from model import SMMGCL
from train import Train
from utils import setup_seed, mclust_R


def construct_sparse_float_tensor(np_matrix):
    """
        construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
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


def loaddata(dataset):
    print("load dataset")
    path = "../generate_data/" + dataset + "/"
    adata_omics1 = sc.read_h5ad(path + "adata_omics1.h5ad")
    adata_omics2 = sc.read_h5ad(path + "adata_omics2.h5ad")
    adata_omics3 = sc.read_h5ad(path + "adata_omics3.h5ad")

    feature_list = []
    spatial_list = []
    adj_wave_list = []
    adj_hat_list = []
    labels_list = []

#################################################
    feature = torch.from_numpy(adata_omics1.X).float()
    feature_list.append(feature)
    spatial_list.append(adata_omics1.obsm['spatial'])
    graph_dict = np.load(path + str(0) + "_graph_dict.npy", allow_pickle=True).tolist()
    adj_hat = graph_dict["adj_hat"]
    adj_wave = graph_dict["adj_wave"]
    adj_hat = construct_sparse_float_tensor(adj_hat)
    adj_hat_list.append(adj_hat)
    adj_wave = construct_sparse_float_tensor(adj_wave)
    adj_wave_list.append(adj_wave)
    labels_list.append(adata_omics1.obs['labels'])
########################################
    feature = torch.from_numpy(adata_omics2.X).float()
    feature_list.append(feature)
    spatial_list.append(adata_omics2.obsm['spatial'])
    graph_dict = np.load(path + str(1) + "_graph_dict.npy", allow_pickle=True).tolist()
    adj_hat = graph_dict["adj_hat"]
    adj_wave = graph_dict["adj_wave"]
    adj_hat = construct_sparse_float_tensor(adj_hat)
    adj_hat_list.append(adj_hat)
    adj_wave = construct_sparse_float_tensor(adj_wave)
    adj_wave_list.append(adj_wave)
    labels_list.append(adata_omics2.obs['labels'])
########################################
    feature = torch.from_numpy(adata_omics3.X).float()
    feature_list.append(feature)
    spatial_list.append(adata_omics3.obsm['spatial'])
    graph_dict = np.load(path + str(2) + "_graph_dict.npy", allow_pickle=True).tolist()
    adj_hat = graph_dict["adj_hat"]
    adj_wave = graph_dict["adj_wave"]
    adj_hat = construct_sparse_float_tensor(adj_hat)
    adj_hat_list.append(adj_hat)
    adj_wave = construct_sparse_float_tensor(adj_wave)
    adj_wave_list.append(adj_wave)
    labels_list.append(adata_omics3.obs['labels'])
    return adata_omics1, adata_omics2, adata_omics3,feature_list, spatial_list, adj_wave_list, adj_hat_list, labels_list


if __name__ == '__main__':
    # Configuration settings
    params = {'num_epochs': 100, 'num_clusters': 6, 'rg_weight': 0.1, 'cl_weight': 0.01, 'con_weight': 0.1,
              'fusion_type': 'att', 'hidden_dims': [128, 32],}
    print('params', params)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=str, default=100, help='seed')
    parser.add_argument('--weight_decay', type=float, default=1e-05, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='RMSprop', help='The optimizer type (RMSprop/Adam)')
    parser.add_argument('--lr', type=float, default=1e-05, help='learning rate')

    parser.add_argument('--fusion_type', type=str, default=params['fusion_type'], help='fusion method')
    parser.add_argument('--rg_weight', type=float, default=params['rg_weight'], help='weight of re graph loss')
    parser.add_argument('--cl_weight', type=float, default=params['cl_weight'], help='weight of cl loss')
    parser.add_argument('--con_weight', type=float, default=params['con_weight'], help='weight of con loss')
    parser.add_argument('--num_epochs', type=int, default=params['num_epochs'], help='number of training epochs')
    parser.add_argument('--num_clusters', type=int, default=params['num_clusters'], help='number of clusters.')
    parser.add_argument('--hidden_dims', type=int, default=params['hidden_dims'], help='the dim in PM')

    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='0', help='The number of cuda device.')
    parser.add_argument('--dataset', type=str, help='dataset name')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    setup_seed(args.seed)
    datasets = ['Simulation_3']
    for i in range(len(datasets)):
        dataset = datasets[i]
        args.dataset = dataset
        print(dataset)
        args.savepath = './result/' + dataset + '/'
        if not os.path.exists(args.savepath):
            os.mkdir(args.savepath)
        os.environ['R_HOME'] = '/usr/lib/R'

        adata1, adata2, adata3, feature_list, spatial_list, adj_wave_list, adj_hat_list, labels_list = loaddata(dataset)
        args.n = feature_list[0].shape[0]
        args.label = np.array(labels_list[0])
        args.num_classes = 5
        args.num_views = len(feature_list)
        args.view_dims = []
        for j in range(args.num_views):
            args.view_dims.append(feature_list[j].shape[1])

        setup_seed(args.seed)
        model = SMMGCL(args.view_dims, args.hidden_dims, args.num_classes, args.fusion_type)
        embedding = Train(model=model,
                          feature_list=feature_list.copy(),
                          adj_hat_list=adj_hat_list.copy(),
                          adj_wave_list=adj_wave_list.copy(), args=args)

        adata1.obsm['embedding'] = embedding
        adata1 = mclust_R(adata1, num_cluster=args.num_classes)
        mclust = np.array(adata1.obs['mclust']).astype(int)

        from ops_ev import get_evaluation_results
        ACC, NMI, ARI = get_evaluation_results(args.label, mclust)
        ACC = round(ACC * 100)
        NMI = round(NMI * 100)
        ARI = round(ARI * 100)
        print('Mclust    ACC: ', ACC, '  NMI: ', NMI, '  ARI: ', ARI)

        # # visualization
        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('Agg')
        #
        # plt.rcParams['figure.figsize'] = (3.5, 3)
        # sc.pl.embedding(adata1, basis='spatial', color='mclust',  # ax=ax_list[1],
        #                 title='embedding', s=100, show=False)
        # plt.tight_layout(w_pad=0.3)
        # plt.show()
        # plt.savefig(args.savepath + 'SMMGCL_mclust.jpg', bbox_inches='tight', dpi=300)
