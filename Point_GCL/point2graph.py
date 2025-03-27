import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
import Point_GCL.GCL.augmentors as A
import Point_GCL.GCL.losses as L
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from torch.optim import Adam
from Point_GCL.GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool


def load_point_cloud_from_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points)
    return point_cloud


def divide_point_cloud(point_cloud, nodes_per_graph=20):
    sub_pcs = [point_cloud[i:i + nodes_per_graph] for i in range(0, len(point_cloud), nodes_per_graph)]
    if len(point_cloud) % nodes_per_graph != 0:
        sub_pcs.append(point_cloud[-(len(point_cloud) % nodes_per_graph):])
    return sub_pcs


def point_cloud_to_graph(sub_pc, k_neighbors=20):
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(sub_pc))).fit(sub_pc)
    _, indices = nbrs.kneighbors(sub_pc)
    G = nx.Graph()
    for idx, pos in enumerate(sub_pc):
        G.add_node(idx, pos=pos)
    G.add_nodes_from(range(len(sub_pc)))
    for idx, neighbors in enumerate(indices):
        G.add_edges_from([(idx, int(neighbor)) for neighbor in neighbors[1:]])
    return G


def graph_to_data(G):
    node_features = np.array([list(G.nodes[n]['pos']) for n in G.nodes])
    num_features = node_features.shape[1]

    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = Data(x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index)

    data.num_nodes = len(G.nodes)
    data.num_features = num_features

    return data


def process_point_cloud(point_cloud, nodes_per_graph=20, k_neighbors=20):

    sub_pcs = divide_point_cloud(point_cloud, nodes_per_graph=nodes_per_graph)

    dataset = [graph_to_data(point_cloud_to_graph(pc, k_neighbors=k_neighbors)) for pc in sub_pcs]

    return dataset


def visualize_graph(G):

    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='red', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edge_color='green', alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

    plt.title('Graph Representation of Point Cloud')
    plt.axis('off')
    plt.show()


class CustomGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cpu')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def main():
    point_cloud = load_point_cloud_from_pcd(".../point_cloud/xxx 10000.pcd")
    print("point_cloud", point_cloud.shape)

    sub_pcs = divide_point_cloud(point_cloud)
    graphs = [point_cloud_to_graph(pc) for pc in sub_pcs]
    print(f"Total {len(graphs)} Graphs")

    for i, G_data in enumerate(graphs[:2]):
        if isinstance(G_data, Data):
            G = nx.Graph(edge_index=G_data.edge_index.numpy().T.tolist())
        else:
            G = G_data
        print(f"Graph {i + 1}'s nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        visualize_graph(G)

    processed_dataset = process_point_cloud(point_cloud, nodes_per_graph=20, k_neighbors=20)
    graph_dataset = CustomGraphDataset(processed_dataset)
    dataloader = DataLoader(graph_dataset, batch_size=2, shuffle=True)

    def get_max_num_features(dataset):
        return max(data.num_features for data in dataset if hasattr(data, 'num_features'))

    input_dim = get_max_num_features(graph_dataset)
    print("input_dim", input_dim)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)

    gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2))
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G')

    optimizer = Adam(encoder_model.parameters(), lr=0.0001)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

        torch.save(encoder_model.state_dict(), 'encoder_xxx.pth')
        print("weight has been saved")


if __name__ == '__main__':
    main()
