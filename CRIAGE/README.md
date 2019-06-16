## Installation

1. Install [PyTorch (version 0.3.0.post4)](http://pytorch.cn/previous-versions/).
2. Install the requirements `pip install -r requirements`


## Running the model

Choosing the link prediction method and the dataset, to run the model for injecting the attacks, you just need to run the following bash file: 
 > bash run.sh inject DistMult WN-18 True 

To run the model for removing the attacks you just need to run the following bash file: 
 > bash run.sh remove DistMult WN-18 True 


# Dependency

The link prediction model is based on ConvE implementation from [Convolutional 2D Knowledge Graph Embeddings](https://github.com/TimDettmers/ConvE)

