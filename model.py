import torch.nn as nn
import torch.nn.functional as F
from layers import GCNConv_dense

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv_dense(in_channels, hidden_channels))
        for i in range(num_layers - 2):
            self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
        self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.sparse = sparse

    def forward(self, x):

        if self.sparse:
            Adj = self.Adj
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(self.Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x