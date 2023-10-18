# SolvingHardAnalogyQuestions

This repo provides the source code & data of our paper titled "Solving Hard Analogy Questions with Relation Embedding Chains", presented at EMNLP 2023.

## Citation
If you use this work in your research, please cite our paper:

```
@InProceedings{kumarn8emnlp,
  author    = {Nitesh Kumar and Steven Schockaert},
  title     = {Solving Hard Analogy Questions with Relation Embedding Chains},
  year      = {2023},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
}
```

## Setup Instructions

### Installation:

```commandline
conda create -n analogies python=3.10.11
conda activate analogies
pip install relbert
pip install unidecode
pip install gensim
pip install git+https://github.com/yuce/pyswip@ab3a36d#egg=pyswip
pip install lmdb
pip install redis
conda install -c conda-forge rocksdb
conda install -c conda-forge python-rocksdb
```

### Download resources:
*  Download the "checkpoint" folder containing trained models and additional files from [this link](https://cf-my.sharepoint.com/:f:/g/personal/kumarn8_cardiff_ac_uk/EqnafbhDt-pMpnroAM_H4GYBfOp6eGCzis_riCFrc1ZyXA?e=6lGWi7)
*  Move the downloaded folder to the appropriate location:

```commandline
sudo mkdir -p /scratch/c.scmnk4/elexir/
chmod 700 /scratch/c.scmnk4/elexir/
mv checkpoint /scratch/c.scmnk4/elexir/resources
```

