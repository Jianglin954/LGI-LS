# LGI-LS (NeurIPS 2023)
Codes for the paper `Latent Graph Inference with Limited Supervision`.

## Datasets

The `Cora`, `Citeseer`, and `Pubmed` datasets can be download from  [here](https://github.com/tkipf/gcn/tree/master/gcn/data). Please place the downloaded files in the folder `data_tf`. The `ogbn-arxiv` dataset will be loaded automatically.

## Dependencies

PLease refer to [SLAPS](https://github.com/BorealisAI/SLAPS-GNN/tree/main) for details.

## Usage

We provide `GCN+KNN`, `GCN+KNN_U`, and `GCN+KNN_R` as examples due to their simpilicity and effectiveness. To test their performances on the `Pubmed` dataset, run the following command:

```bash
bash experiments.sh
```

The experimental results will be saved in the corresponding *.txt file.

# Reference

    @inproceedigs{LGI-LS,
      title={Latent Graph Inference with Limited Supervision},
      author={Jianglin Lu, Yi Xu, Huan Wang, Yue Bai, Yun Fu},
      booktitle={Advances in Neural Information Processing Systems},
      year={2023}
    }

# Acknowledgement
Our codes are mainly based on [SLAPS](https://github.com/BorealisAI/SLAPS-GNN/tree/main). For other comparison methods, please refer to their publicly available code repositories. We gratefully thank the authors for their contributions. 
