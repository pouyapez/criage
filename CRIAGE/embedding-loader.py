import torch
import pickle
import numpy as np
import torch.backends.cudnn as cudnn

from model import ConvE

model = ConvE
#the_model = TheModelClass(*args, **kwargs)

model.load_state_dict(torch.load('embeddings.pt'))

print len(the_model['emb_e'])