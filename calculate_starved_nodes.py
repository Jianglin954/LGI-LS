import torch
import argparse
from citation_networks import load_citation_network_calculate_starved_nodes
import numpy as np


def Calculate_Number_of_Starved_Nodes(args):
    features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj = load_citation_network_calculate_starved_nodes(args.dataset)

    #######  Citeseer 370  #######
    ## train_mask[120:370] = True
    ## val_mask[120:370] = False
    #######   Cora 390    #######
    ## train_mask[140:390] = True
    ## val_mask[140:390] = False

    adj = adj + np.eye(adj.shape[0])
    adj = torch.from_numpy(adj)
    adj1hop = adj[:, train_mask]
    adj1hop = adj1hop.sum(dim=1)
    adj1hop = adj1hop.numpy()
    connect_label = np.sum(adj1hop > 0)
    num_of_1hop_starved_nodes = adj1hop.shape[0] - connect_label
    notcon_label_index = np.where(adj1hop == 0)[0]

    adjpower2 = torch.mm(adj, adj)
    adjpower2np = adjpower2.numpy()
    index = np.argwhere(adjpower2np == 1)
    rows = index[:, 0]
    rows = np.unique(rows)
    label_adjpower2np = adjpower2np[:, train_mask]
    label_index = np.argwhere(label_adjpower2np == 1)
    label_rows = label_index[:, 0]
    label_rows = np.unique(label_rows)
    diff_rows = np.setdiff1d(rows, label_rows)
    interrows = np.intersect1d(notcon_label_index, diff_rows)
    num_of_2hop_starved_nodes = interrows.size

    adjpower3 = torch.mm(adjpower2, adj)
    adjpower3np = adjpower3.numpy()
    index3hop = np.argwhere(adjpower3np == 1)
    rows3ho = index3hop[:, 0]
    rows3ho = np.unique(rows3ho)
    label_adjpower3np = adjpower3np[:, train_mask]
    label_index3hop = np.argwhere(label_adjpower3np == 1)
    label_rows3hop = label_index3hop[:, 0]
    label_rows3hop = np.unique(label_rows3hop)
    diff_rows3hop = np.setdiff1d(rows3ho, label_rows3hop)
    interrows3hop = np.intersect1d(interrows, diff_rows3hop)
    num_of_3hop_starved_nodes = interrows3hop.size

    adjpower4 = torch.mm(adjpower3, adj)
    adjpower4np = adjpower4.numpy()
    index4hop = np.argwhere(adjpower4np == 1)
    rows4ho = index4hop[:, 0]
    rows4ho = np.unique(rows4ho)
    label_adjpower4np = adjpower4np[:, train_mask]
    label_index4hop = np.argwhere(label_adjpower4np == 1)
    label_rows4hop = label_index4hop[:, 0]
    label_rows4hop = np.unique(label_rows4hop)
    diff_rows4hop = np.setdiff1d(rows4ho, label_rows4hop)
    interrows4hop = np.intersect1d(interrows3hop, diff_rows4hop)
    num_of_4hop_starved_nodes = interrows4hop.size

    print("The number of 1-, 2-, 3-, and 4-hop starved nodes on " + args.dataset +
          " dataset are {:0d}, {:0d}, {:0d}, and {:0d}, respectively.".format(num_of_1hop_starved_nodes,
          num_of_2hop_starved_nodes, num_of_3hop_starved_nodes, num_of_4hop_starved_nodes))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, default='cora', help='See choices',
                        choices=['cora', 'citeseer', 'pubmed'])

    args = parser.parse_args()
    Calculate_Number_of_Starved_Nodes(args)
