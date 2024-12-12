import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils import dot_product_decode, all_fg_consstruct, build_layer_units
from typing import List


class SMMGCL(nn.Module):

    def __init__(self, view_dims, hidden_dims, num_clusters,fusion_type):
        super(SMMGCL, self).__init__()
        self.view_dims = view_dims
        self.hidden_dims = hidden_dims
        self.num_views = len(view_dims)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(self.num_views):
            temp_dims = []
            temp_dims.append(view_dims[i])
            temp_dims.extend(self.hidden_dims)
            self.encoder.append(MLFPN_GCN(temp_dims, nn.ReLU()))
            self.decoder.append(MLFPN_FC(list(reversed(temp_dims)), nn.ReLU()))

        aemp_dims = []
        aemp_dims.extend([self.hidden_dims[-1],self.hidden_dims[-1]])
        self.full_graph_module = MLFPN_GCN(aemp_dims, nn.ReLU())
        self.fusion_module = FusionLayer(num_views=self.num_views, hidden_dim=self.hidden_dims[-1],fusion_type=fusion_type)
        self.cluster_layer = ClusterLayer(num_clusters, self.hidden_dims[-1])

    def forward(self, feats, adjs):
        hidden_pr_list = []
        X_bar_list = []
        adj_bar_list = []
        Xz_list = []

        for i in range(self.num_views):
            hidden_pr = self.encoder[i](feats[i], adjs[i])
            hidden_pr_list.append(hidden_pr)
            X_bar = self.decoder[i](hidden_pr)
            X_bar_list.append(X_bar)
            adj_bar = dot_product_decode(hidden_pr)
            adj_bar_list.append(adj_bar)

        z, weight_z = self.fusion_module(hidden_pr_list)

        adj_all = all_fg_consstruct(hidden_pr_list, adj_bar_list, self.num_views)
        hidden_tensor_he = torch.stack(hidden_pr_list, 0)
        hidden_tensor_he = hidden_tensor_he.reshape(-1, self.hidden_dims[-1])
        hidden_tensor_all = self.full_graph_module(hidden_tensor_he, adj_all)
        hidden_tensor_all = hidden_tensor_all.reshape(self.num_views, -1, self.hidden_dims[-1])

        hidden_list_all = []
        for i in range(self.num_views):
            hidden_list_all.append(hidden_tensor_all[i])

        h, weight_h = self.fusion_module(hidden_list_all)

        adjz = dot_product_decode(z)

        for i in range(self.num_views):
            X_bar = self.decoder[i](z)
            Xz_list.append(X_bar)
        qz = self.cluster_layer(z)
        qh = self.cluster_layer(h)
        return h, z, adjz, Xz_list, qz, qh


class MLFPN_GCN(nn.Module):

    def __init__(self, dims: List[int], act_func: nn.Module=nn.ReLU()):
        super(MLFPN_GCN, self).__init__()
        self.network = build_layer_units(layer_type='gcn', dims=dims, act_func=act_func)

    def forward(self, fea: torch.Tensor, adj: torch.sparse) -> torch.Tensor:
        output = fea
        for seq in self.network:
            if len(seq) == 1:
                output = seq[0](output, adj)
            elif len(seq) == 2:
                output = seq[0](output, adj)
                output = seq[1](output)
        return output


class MLFPN_FC(nn.Module):

    def __init__(self, dims: List[int], act_func: nn.Module = nn.ReLU()):
        super(MLFPN_FC, self).__init__()
        self.network = build_layer_units(layer_type='linear', dims=dims, act_func=act_func)

    def forward(self, fea: torch.Tensor) -> torch.Tensor:
        output = fea
        for seq in self.network:
            if len(seq) == 1:
                output = seq[0](output)
            elif len(seq) == 2:
                output = seq[0](output)
                output = seq[1](output)
        return output


class FusionLayer(nn.Module):

    def __init__(self, num_views=2,hidden_dim=32, fusion_type='att'):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        self.num_views = num_views
        self.att = Attention(hidden_dim)

    def forward(self, features):
        combined_feature=[]
        if self.fusion_type == "weight":
            combined_feature = 1/self.num_views * features[0]
            for i in range(1, self.num_views):
                combined_feature = combined_feature + 1/self.num_views * features[i]
            weight = torch.ones(features[0].shape[0], self.num_views) * 1/self.num_views
        elif self.fusion_type == "att":
            combined_feature = features[0].unsqueeze(1)  # 增加一个维度，使大小变为 [1296, 1, 64]
            for i in range(1, self.num_views):
                combined_feature = torch.cat([combined_feature, features[i].unsqueeze(1)], dim=1)  # 沿着第二个维度拼接，保持大小一致
            combined_feature, weight = self.att(combined_feature)
        return combined_feature, weight


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class ClusterLayer(nn.Module):

    def __init__(self, num_clusters, hidden_dim, alpha=1):
        super(ClusterLayer, self).__init__()
        self.alpha = alpha
        self.network = Parameter(torch.Tensor(num_clusters, hidden_dim)).float()
        torch.nn.init.xavier_normal_(self.network.data)

    def forward(self, z) -> torch.Tensor:
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.network, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q