import torch.nn as nn
from torch_geometric.nn import GCN, global_mean_pool


class GCNModel(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.6
    ):
        super().__init__()
        self.gnc_model = GCN(
            in_channels, hidden_channels, num_layers, dropout=dropout, jk="cat"
        )
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        out = self.gnc_model(x, edge_index, edge_weight=edge_weight, batch=batch)
        # print("GCN output shape: ", out.shape)
        out = global_mean_pool(out, batch)  # [batch_size, hidden_channels]
        # print("global mean pooling output shape: ", out.shape)
        out = self.softmax(self.fc(out))
        # print("softmax output shape: ", out.shape)
        return out
