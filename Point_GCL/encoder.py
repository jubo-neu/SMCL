import torch
import torch.nn as nn
from tqdm import tqdm
import Point_GCL.GCL.augmentors as A
import Point_GCL.GCL.losses as L

from torch_geometric.data import Data, DataLoader
from tqdm.auto import tqdm
from Point_GCL.GCL.models import DualBranchContrast
from .point2graph import (
    load_point_cloud_from_pcd,
    divide_point_cloud,
    point_cloud_to_graph,
    graph_to_data,
    Encoder,
    GConv
)


class PointEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, pretrained=False, pretrained_weights_path=None):
        super().__init__()

        self.input_dim = input_dim

        aug1 = A.Identity()
        aug2 = A.RandomChoice([
                               A.NodeDropping(pn=0.1),
                               A.FeatureMasking(pf=0.1),
                               A.EdgeRemoving(pe=0.1)], 1)
        gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2))
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G')

        if pretrained:
            assert pretrained_weights_path is not None
            self.load_pretrained_weights(pretrained_weights_path)
            self.encoder_model.eval()

    def load_pretrained_weights(self, pretrained_weights_path):
        state_dict = torch.load(pretrained_weights_path, map_location='cpu')
        self.encoder_model.load_state_dict(state_dict)

    def point_cloud_to_data_loader(self, point_cloud_file, nodes_per_graph=20, k_neighbors=20):
        point_cloud = load_point_cloud_from_pcd(point_cloud_file)
        sub_pcs = divide_point_cloud(point_cloud, nodes_per_graph)
        graphs = [point_cloud_to_graph(pc, k_neighbors) for pc in sub_pcs]
        datasets = [graph_to_data(G) for G in graphs]
        return DataLoader(datasets, batch_size=1, shuffle=False)

    def forward(self, point_cloud_file):
        data_loader = self.point_cloud_to_data_loader(point_cloud_file)

        for data in tqdm(data_loader, desc='Encoding Batches'):
            z, g, z1, z2, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)

        return z
