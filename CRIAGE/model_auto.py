import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from spodernet.utils.global_config import Config
from spodernet.utils.cuda_utils import CUDATimer
from torch.nn.init import xavier_normal, xavier_uniform
from spodernet.utils.cuda_utils import CUDATimer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

timer = CUDATimer()


class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal(self.emb_e_real.weight.data)
        xavier_normal(self.emb_e_img.weight.data)
        xavier_normal(self.emb_rel_real.weight.data)
        xavier_normal(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.inp_drop(self.emb_e_real(e1)).view(Config.batch_size, -1)
        rel_embedded_real = self.inp_drop(self.emb_rel_real(rel)).view(Config.batch_size, -1)
        e1_embedded_img = self.inp_drop(self.emb_e_img(e1)).view(Config.batch_size, -1)
        rel_embedded_img = self.inp_drop(self.emb_rel_img(rel)).view(Config.batch_size, -1)

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_e.weight.requires_grad=False 
        self.emb_rel.weight.requires_grad=False 
        self.linear_t = torch.nn.Linear(Config.embedding_dim, Config.embedding_dim)
        self.linear_rel = torch.nn.Linear(Config.embedding_dim, num_relations)
        self.linear_e1 = torch.nn.Linear(Config.embedding_dim, num_entities)
        self.linear_t.weight.requires_grad=True
        self.linear_e1.weight.requires_grad=True
        self.linear_rel.weight.requires_grad=True
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.CrossEntropyLoss()

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.view(-1, Config.embedding_dim)
        rel_embedded = rel_embedded.view(-1, Config.embedding_dim)

        pred = e1_embedded*rel_embedded
	return self.decoder(pred)

    def encoder(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.view(-1, Config.embedding_dim)
        rel_embedded = rel_embedded.view(-1, Config.embedding_dim)

        pred = e1_embedded*rel_embedded
	return pred

    def encoder_2(self, e1):
        e1_embedded= self.emb_e(e1)
	return e1_embedded

    def decoder(self, pred):
        pred = self.linear_t(pred)
        pred= F.relu(pred)
        E1 = self.linear_e1(pred)
        R = self.linear_rel(pred)
        return E1, R
	 

class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_e.weight.requires_grad=False 
        self.emb_rel.weight.requires_grad=False 
        self.linear_t = torch.nn.Linear(Config.embedding_dim, 10368)
        self.linear_rel = torch.nn.Linear(2*Config.embedding_dim, num_relations)
        self.linear_e1 = torch.nn.Linear(2*Config.embedding_dim, num_entities)
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=3)
        self.linear_t.weight.requires_grad=True
        self.linear_e1.weight.requires_grad=True
        self.linear_rel.weight.requires_grad=True
        self.deconv1.weight.requires_grad=True
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.CrossEntropyLoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368,Config.embedding_dim)
        self.conv1.weight.requires_grad=False 
        self.fc.weight.requires_grad=False 
        print(num_entities, num_relations)

    def init(self):
        xavier_normal(self.emb_e.weight.data)
        xavier_normal(self.emb_rel.weight.data)

    def forward(self, e1, rel, E2= None):
        e1_embedded= self.emb_e(e1).view(Config.batch_size, 1, 10, 20)#(Config.batch_size, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).view(Config.batch_size, 1, 10, 20)#(Config.batch_size, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = stacked_inputs
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = x.view(Config.batch_size, -1)
        x = self.fc(x)
        x = self.bn2(x)
        x = F.relu(x)

        return self.decoder(x)

    def encoder(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = stacked_inputs 
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = x.view(1, 10368)
        x = self.fc(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

    def encoder_2(self, e1):
        e1_embedded= self.emb_e(e1)
        return e1_embedded

    def decoder(self, pred):
        pred = self.linear_t(pred).view(-1, 32, 18, 18)
        pred = self.deconv1(pred)

        pred= F.relu(pred.view(-1, 400))
        E1 = self.linear_e1(pred)
        R = self.linear_rel(pred)
        return E1, R


