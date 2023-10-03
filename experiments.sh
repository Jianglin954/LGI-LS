#!/bin/bash

######################        GCN_KNN_R ON Pubmed         #######################
python main.py -dataset pubmed -dropout2 0.5 -dropout_adj2 0.0 -epochs 2000 -half_train 0 \
-half_val_as_train 0  -hidden 32  -knn_metric cosine -lr 0.01 -nlayers 2 \
-normalization sym  -ntrials 5  -patience 3000  -sparse 0 -w_decay 0.0005 \
-k 15  -alpha 100 -klabel 30  | tee -a GCN_KNN_R.txt
