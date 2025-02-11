# On the Expressive Power of Subgraph Graph Neural Networks for Graphs with Bounded Cycles

This repository is the official implementation of "[On the Expressive Power of Subgraph Graph Neural Networks for Graphs with Bounded Cycles](https://arxiv.org/abs/2502.03703)", forked from [Graphormer-GD](https://github.com/lsj2408/Graphormer-GD), based on the official implementation of [Graphormer](https://github.com/microsoft/Graphormer) in [PyTorch](https://github.com/pytorch/pytorch).

## Overview

## Installation

- Clone this repository

```shell
git clone https://github.com/rogerwwww/k-hop-Graphormer.git
```

- Install the dependencies (Using [Anaconda](https://www.anaconda.com/), tested with CUDA version 11.1)

```shell
# upgrade gcc & gxx versions
conda install -y gcc gxx

# install requirements
pip install pip==24.0
pip install numpy==1.22.4
pip install torch==1.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
pip install lmdb
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-geometric==1.7.2
pip install tensorboardX==2.4.1
pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
pip install protobuf==3.20.3

cd engine
pip install -e .
python setup.py build_ext --inplace
# you may have to add #include <cstdint> in pythonpath/lib/python3.9/site-packages/torch/include/pybind11/attr.h
```

## Training k-hop Graphormers
**On Graph Representation Learning (ZINC dataset)**

With Graphormer backbone (**k-hop Graphormer**):

``. examples/k-hop-Graphormer/train_k_hop_graphormer_zinc.sh``

With Graphormer-GD backbone (**k-hop Graphormer-GD**):

``. examples/k-hop-Graphormer/train_k_hop_graphormer_gd_zinc.sh``

**Some practical notes:**
1. Set ``subgraph_radius`` to test with different ``k``s on k-hop Graphormer
2. When running multiple training jobs on the same node, please set different ``MASTER_PORT`` for different jobs
3. You shall be able to collect results, training and testing statistics with tensorboard

## Citation

If you find this work useful, please kindly cite following papers:

**k-hop Graphormer**
```latex
@article{chen2025expressive,
  title={On the Expressive Power of Subgraph Graph Neural Networks for Graphs with Bounded Cycles},
  author={Chen, Ziang and Zhang, Qiao and Wang, Runzhong},
  journal={arXiv preprint arXiv:2502.03703},
  year={2025}
}
```

**Graphormer-GD**
```latex
@inproceedings{zhang2023rethinking,
  title={Rethinking the Expressive Power of {GNN}s via Graph Biconnectivity},
  author={Bohang Zhang and Shengjie Luo and Liwei Wang and Di He},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=r9hNv76KoT3}
}
```

**Graphormer**
```latex
@article{ying2021transformers,
  title={Do transformers really perform badly for graph representation?},
  author={Ying, Chengxuan and Cai, Tianle and Luo, Shengjie and Zheng, Shuxin and Ke, Guolin and He, Di and Shen, Yanming and Liu, Tie-Yan},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={28877--28888},
  year={2021}
}
```

## License

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/lsj2408/Transformer-M/blob/main/LICENSE) for additional details.
