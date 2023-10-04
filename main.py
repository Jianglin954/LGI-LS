import argparse
import random
from data_loader import load_data
from citation_networks import load_citation_network_halftrain
from model import GCN
from utils import *

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()

    def get_loss_fixed_adj(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def half_val_as_train(self, val_mask, train_mask):
        val_size = np.count_nonzero(val_mask)
        counter = 0
        for i in range(len(val_mask)):
            if val_mask[i] and counter < val_size / 2:
                counter += 1
                val_mask[i] = False
                train_mask[i] = True
        return val_mask, train_mask


    def GCN_KNN(self, args):

        if args.half_train:
            print("Using half of labeled nodes for training!")
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_citation_network_halftrain(args.dataset)
        else:
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_data(args)

        Adj = torch.from_numpy(nearest_neighbors(features, args.k, args.knn_metric)).cuda()
        Adj = normalize(Adj, args.normalization, args.sparse)

        if torch.cuda.is_available():
            features = features.cuda()

        if args.half_val_as_train:
            val_mask, train_mask = self.half_val_as_train(val_mask, train_mask)


        test_accu = []
        for trial in range(args.ntrials):
            model = GCN(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses, num_layers=args.nlayers,
                        dropout=args.dropout2, dropout_adj=args.dropout_adj2, Adj=Adj, sparse=args.sparse)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            best_test_accu = 0
            counter = 0
            if torch.cuda.is_available():
                model = model.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()

            for epoch in range(1, args.epochs + 1):
                model.train()
                loss, accu = self.get_loss_fixed_adj(model, train_mask, features, labels)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if epoch % 1 == 0:
                    with torch.no_grad():
                        model.eval()
                        test_loss_, test_accu_ = self.get_loss_fixed_adj(model, test_mask, features, labels)
                        if epoch % 100 == 0:
                            print("Epoch {:04d}: Test Loss {:.4f}, Test Accuracy {:.4f}".format(epoch, test_loss_, test_accu_))
                        if test_accu_ > best_test_accu:
                            counter = 0
                            best_test_accu = test_accu_
                        else:
                            counter += 1
                        if counter >= args.patience:
                            break

            with torch.no_grad():
                model.eval()
                print("*******************************")
                print("Trial {:02d}: test accuracy {:.4f}".format(trial, best_test_accu))
                print("*******************************")
                test_accu.append(best_test_accu.item())

        print(test_accu)
        print("std of test accuracy", np.std(test_accu) * 100)
        print("average of test accuracy", np.mean(test_accu) * 100)


    def GCN_KNN_U(self, args):

        if args.half_train:
            print("Using half of labeled nodes for training!")
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_citation_network_halftrain(args.dataset)
        else:
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_data(args)

        Adj = torch.from_numpy(nearest_neighbors(features, args.k, args.knn_metric)).cuda()



        if torch.cuda.is_available():
            features = features.cuda()

        if args.half_val_as_train:
            val_mask, train_mask = self.half_val_as_train(val_mask, train_mask)



        CUR_C = Adj.clone()[:, train_mask]
        row_mask = torch.sum(CUR_C, dim=1) > 0
        asy_similarities = cal_similarity_graph(features, features[train_mask, :])
        asy_similarities = top_k(asy_similarities, args.klabel)
        asy_similarities[train_mask, :] = 0.0   # Empirically, we found that train_mask works better!!
        # asy_similarities[row_mask, :] = 0.0


        Adj[:, train_mask] = Adj[:, train_mask] + args.alpha * asy_similarities
        Adj = normalize(Adj, args.normalization, args.sparse)

        test_accu = []
        for trial in range(args.ntrials):
            model = GCN(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses, num_layers=args.nlayers,
                        dropout=args.dropout2, dropout_adj=args.dropout_adj2, Adj=Adj, sparse=args.sparse)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            best_test_accu = 0
            counter = 0
            if torch.cuda.is_available():
                model = model.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()

            for epoch in range(1, args.epochs + 1):
                model.train()
                loss, accu = self.get_loss_fixed_adj(model, train_mask, features, labels)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if epoch % 1 == 0:
                    with torch.no_grad():
                        model.eval()
                        test_loss_, test_accu_ = self.get_loss_fixed_adj(model, test_mask, features, labels)
                        if epoch % 100 == 0:
                            print("Epoch {:04d}: Test Loss {:.4f}, Test Accuracy {:.4f}".format(epoch, test_loss_, test_accu_))
                        if test_accu_ > best_test_accu:
                            counter = 0
                            best_test_accu = test_accu_
                        else:
                            counter += 1
                        if counter >= args.patience:
                            break

            with torch.no_grad():
                model.eval()
                print("*******************************")
                print("Trial {:02d}: test accuracy {:.4f}".format(trial, best_test_accu))
                print("*******************************")
                test_accu.append(best_test_accu.item())

        print(test_accu)
        print("std of test accuracy", np.std(test_accu) * 100)
        print("average of test accuracy", np.mean(test_accu) * 100)



    def GCN_KNN_R(self, args):

        if args.half_train:
            print("Using half of labeled nodes for training!")
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_citation_network_halftrain(args.dataset)
        else:
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_data(args)

        Adj = torch.from_numpy(nearest_neighbors(features, args.k, args.knn_metric)).cuda()



        if torch.cuda.is_available():
            features = features.cuda()

        if args.half_val_as_train:
            val_mask, train_mask = self.half_val_as_train(val_mask, train_mask)

        asy_similarities = cal_similarity_graph(features, features[train_mask, :])
        asy_similarities = top_k(asy_similarities, args.klabel)

        Adj[:, train_mask] = Adj[:, train_mask] + args.alpha * asy_similarities
        Adj = normalize(Adj, args.normalization, args.sparse)

        test_accu = []
        for trial in range(args.ntrials):
            model = GCN(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses, num_layers=args.nlayers,
                        dropout=args.dropout2, dropout_adj=args.dropout_adj2, Adj=Adj, sparse=args.sparse)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            best_test_accu = 0
            counter = 0
            if torch.cuda.is_available():
                model = model.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()

            for epoch in range(1, args.epochs + 1):
                model.train()
                loss, accu = self.get_loss_fixed_adj(model, train_mask, features, labels)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if epoch % 1 == 0:
                    with torch.no_grad():
                        model.eval()
                        test_loss_, test_accu_ = self.get_loss_fixed_adj(model, test_mask, features, labels)
                        if epoch % 100 == 0:
                            print("Epoch {:04d}: Test Loss {:.4f}, Test Accuracy {:.4f}".format(epoch, test_loss_, test_accu_))
                        if test_accu_ > best_test_accu:
                            counter = 0
                            best_test_accu = test_accu_
                        else:
                            counter += 1
                        if counter >= args.patience:
                            break


            with torch.no_grad():
                model.eval()
                print("Trial {:02d}: test accuracy {:.4f}".format(trial, best_test_accu))
                test_accu.append(best_test_accu.item())

        print(test_accu)
        print("std of test accuracy", np.std(test_accu) * 100)
        print("average of test accuracy", np.mean(test_accu) * 100)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, default='pubmed', help='See choices',
                        choices=['cora', 'citeseer', 'pubmed', 'ogbn-arxiv'])
    parser.add_argument('-ntrials', type=int, default=5, help='Number of trials')
    parser.add_argument('-epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('-lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('-w_decay', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('-nlayers', type=int, default=2, help='#layers')
    parser.add_argument('-k', type=int, default=15, help='k for initializing with knn')
    parser.add_argument('-knn_metric', type=str, default='cosine', help='See choices', choices=['cosine', 'minkowski'])
    parser.add_argument('-half_val_as_train', type=int, default=0, help='use first half of validation for training')
    parser.add_argument('-half_train', type=int, default=0, help='use half of labeled nodes for training')
    parser.add_argument('-normalization', type=str, default='sym')
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-patience', type=int, default=3000, help='patience for early stopping')
    parser.add_argument('-method', type=str, default='GCN_KNN', help='See choices',
                        choices=['GCN_KNN', 'GCN_KNN_U', 'GCN_KNN_R'])
    experiment = Experiment()


    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    parser.add_argument('-dropout2', type=float, default=0.5, help='Dropout rate in GCN.')
    parser.add_argument('-dropout_adj2', type=float, default=0., help='Dropout rate GCN.')
    parser.add_argument('-alpha', type=float, default=100, help='control the contribution of asy and sys similarity')
    parser.add_argument('-klabel', type=int, default=30, help='k_label for asymmetric similarity')
    args = parser.parse_args()

    if args.method == "GCN_KNN":
        experiment.GCN_KNN(args)
    elif args.method == "GCN_KNN_U":
        experiment.GCN_KNN_U(args)
    elif args.method == "GCN_KNN_R":
        experiment.GCN_KNN_R(args)
    print(args)