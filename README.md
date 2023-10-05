# LGI-LS (NeurIPS 2023)
Codes for the NeurIPS 2023 paper `Latent Graph Inference with Limited Supervision`.

## Datasets

The `Cora`, `Citeseer`, and `Pubmed` datasets can be download from  [here](https://github.com/tkipf/gcn/tree/master/gcn/data). Please place the downloaded files in the folder `data_tf`. The `ogbn-arxiv` dataset will be loaded automatically.


## Installation
```bash
conda create -n LGI python=3.7.2
conda activate LGI
pip install torch==1.5.1 torchvision==0.6.1
pip install scipy==1.2.1
pip install scikit-learn==0.21.3
pip install dgl==0.5.2
pip install ogb==1.2.3
wget https://data.pyg.org/whl/torch-1.5.0%2Bcu102/torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.5.0%2Bcu102/torch_sparse-0.6.5-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.5.0%2Bcu102/torch_cluster-1.5.4-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.5.0%2Bcu102/torch_spline_conv-1.2.0-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.5-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.4-cp37-cp37m-linux_x86_64.whl
pip install torch_spline_conv-1.2.0-cp37-cp37m-linux_x86_64.whl
pip install pytorch-geometric==1.6.1
```


## Usage

We provide `GCN+KNN`, `GCN+KNN_U`, and `GCN+KNN_R` as examples due to their simplicity and effectiveness. To test their performances on the `Pubmed` dataset, run the following command:

```bash
bash experiments.sh
```

The experimental results will be saved in the corresponding *.txt file.

# Reference

    @inproceedigs{Jianglin2023LGI,
      title={Latent Graph Inference with Limited Supervision},
      author={Jianglin Lu, Yi Xu, Huan Wang, Yue Bai, Yun Fu},
      booktitle={Advances in Neural Information Processing Systems},
      year={2023}
    }

    @inproceedigs{fatemi2021slaps,
      title={SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks},
      author={Fatemi, Bahare and Asri, Layla El and Kazemi, Seyed Mehran},
      booktitle={Advances in Neural Information Processing Systems},
      year={2021}
    }

# Acknowledgement
Our codes are mainly based on [SLAPS](https://github.com/BorealisAI/SLAPS-GNN/tree/main). For other comparison methods, please refer to their publicly available code repositories. We gratefully thank the authors for their contributions. 
