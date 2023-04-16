Adversarial-Graph-Node-Classification-Based-On-Unsupervised-Learning-And-Optimized-Loss-Functions

This project is the implementation of the paper "Adversarial-Graph-Node-Classification-Based-On-Unsupervised-Learning-And-Optimized-Loss-Functions".

This repo contains the codes, data and results reported in the paper.

Dependencies
-----

The script has been tested running under Python 3.8.5, with the following packages installed (along with their dependencies):

numpy==1.18.4
scipy==1.4.1
torch==1.4.0
tqdm==4.42.1
networkx==2.4
scikit-learn==0.23.1


Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.

Usage: Model Training
==================================================================
We include all three benchmark datasets Cora, Citeseer and Polblogs in the ```data``` directory.

eg:python train.py --dataset cora --alpha 0.4 --epsilon 0.1 --tau 0.005
      

Usage: Evaluation
-----
We provide the evaluation codes on the node classification task here. 
We evaluate on three real-world datasets Cora, Citeseer and Polblogs. 

eg:python eval.py --dataset cora --alpha 0.2 --epsilon 0.1 --model model.pkl
      
