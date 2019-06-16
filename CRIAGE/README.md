



## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch](https://github.com/pytorch/pytorch) using [Anaconda](https://www.continuum.io/downloads)
2. Install the requirements `pip install -r requirements`
3. Run the preprocessing script for WN18RR, FB15k-237, YAGO3-10, UMLS, Kinship, and Nations: `sh preprocess.sh`
3. You can now run the model




bash run.sh inject DistMult WN-18 True 



# Dependency

The link prediction model is based on ConvE implementation from [Convolutional 2D Knowledge Graph Embeddings](https://github.com/TimDettmers/ConvE)

