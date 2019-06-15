import json
import torch
import pickle
import numpy as np
import numpy
import argparse
import sys
import os
import math
reload(sys)
sys.setdefaultencoding('utf-8')
import codecs
import random

from os.path import join
import torch.backends.cudnn as cudnn

#from num_process import num_nextbatch
from evaluation import ranking_and_hits, attack_tri
from model_auto import ConvE, DistMult, Complex

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
np.set_printoptions(precision=3)

timer = CUDATimer()
cudnn.benchmark = True

# parse console parameters and set global variables
Config.backend = Backends.TORCH
Config.parse_argv(sys.argv)

Config.cuda = True
Config.embedding_dim = 200
#Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG


#model_name = 'DistMult_{0}_{1}'.format(Config.input_dropout, Config.dropout)
model_name = '{2}_{0}_{1}'.format(Config.input_dropout, Config.dropout, Config.model_name)
epochs = 95 
load = False

if Config.dataset is None:
    Config.dataset = 'WN-18'
model_path = 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)


def sig (x, y):
    return 1 / (1 + np.exp(-np.dot(x, np.transpose(y))))
def point_hess(e_o, nei, embd_e, embd_rel):
    H = np.zeros((200, 200))
    for i in nei:
        X = np.multiply(np.reshape(embd_e[i[0]], (1, -1)), np.reshape(embd_rel[i[1]], (1, -1)))
        sig_tri = sig(e_o, X)
        Sig = (sig_tri)*(1-sig_tri)
        H += Sig * np.dot(np.transpose(X), X)
    return H

def point_score(Y, X, e_o, H):
    sig_tri = sig(e_o, X) 
    M = np.linalg.inv(H + (sig_tri)*(1-sig_tri)*np.dot(np.transpose(X), X))
    Score = np.dot(Y, np.transpose((1-sig_tri)*np.dot(X, M)))
    return Score, M

def grad_score(Y, X, H, e_o, M):
    grad = []
    n = 200
    sig_tri = sig(e_o, X)
    A = H + (sig_tri)*(1-sig_tri)*np.dot(np.transpose(X), X)
    A_in = M 
    X_2 = np.dot(np.transpose(X), X)
    f_part = np.dot(Y, np.dot((1-sig_tri)*np.eye(n)-(sig_tri)*(1-sig_tri)*np.transpose(np.dot(np.transpose(e_o), X)), A_in))
    for i in range(n):
        s = np.zeros((n,n))
        s[:,i] = X
        s[i,:] = X
        s[i,i] = 2*X[0][i]
        Q = np.dot(((sig_tri)*(1-sig_tri)**2-(sig_tri)**2*(1-sig_tri))*e_o[0][i]*X_2+(sig_tri)*(1-sig_tri)*s, A_in)
        grad += [f_part[0][i] - np.dot(Y, np.transpose((1-sig_tri)*np.dot(X, np.dot(A_in, Q))))[0][0]] ######## + 0.02 * X[0][i]]

    return grad

def find_best_attack(e_o, Y, nei, embd_e, embd_rel, attack_ext, pr):
    H = point_hess(e_o, nei, embd_e, embd_rel)
    X = pr
    step = np.array([[0.00000000001]])
    score = 0 
    score_orig,_ = point_score(Y, pr, e_o,H)
    score_n, M = point_score(Y, X, e_o,H)
    num_iter = 0
    while score_n >= score_orig or num_iter<1:
        if num_iter ==4:
            X = pr
            break
        num_iter += 1
        Grad = grad_score(Y, X, H, e_o, M)
        X = X + step * Grad 
        score = score_n
        score_n, M = point_score(Y, X, e_o, H)

    return X


def find_best_at(pred, E2):
    e2 = E2.view(-1).data.cpu().numpy()
    Pred = pred.view(-1).data.cpu().numpy()
    A1 = np.dot(Pred, e2)
    A2 = np.dot(e2, e2)
    A3 = np.dot(Pred, Pred)
    A = math.sqrt(np.true_divide(A3*A2-0.2, A3*A2-A1**2))
    B = np.true_divide(math.sqrt(0.2)-A*A1, A2) 
    return float(A), float(B)


''' Preprocess knowledge graph using spodernet. '''
def preprocess(dataset_name, delete_data=False):
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(Config.dataset, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()


    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data #if isinstance(y, torch.Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    #y_one_hot = y_one_hot.view(*(y.shape), -1)
    return torch.autograd.Variable(y_one_hot).cuda() #if isinstance(y, Variable) else y_one_hot

def main():
    if Config.process: preprocess(Config.dataset, delete_data=True)
    input_keys = ['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(Config.dataset, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token
    dict_tokentoid, dict_idtotoken = vocab['e1'].tokendicts()
    dict_reltoid, dict_idtorel = vocab['rel'].tokendicts()

    num_rel = vocab['rel'].num_token 
    train_batcher = StreamBatcher(Config.dataset, 'train', Config.batch_size, randomize=True, keys=input_keys)
    dev_rank_batcher = StreamBatcher(Config.dataset, 'dev_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys, is_volatile=True)
    test_rank_batcher = StreamBatcher(Config.dataset, 'test_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys, is_volatile=True)


    if Config.model_name is None:
        model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'ConvE':
        model = ConvE(vocab['e1'].num_token, num_rel)
    elif Config.model_name == 'DistMult':
        model = DistMult(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'ComplEx':
        model = Complex(vocab['e1'].num_token, vocab['rel'].num_token)
    else:
        log.info('Unknown model: {0}', Config.model_name)
        raise Exception("Unknown model!")

    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))


    eta = ETAHook('train', print_every_x_batches=100)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=100))

    if Config.cuda:
        model.cuda()
    if load:
        model_params = torch.load(model_path)
        print(model)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        model.eval()
        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
        ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
    else:
        model.init()

    total_param_size = []
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(np.sum(params))

################################################ loading
    model.load_state_dict(torch.load('embeddings/auto-embeddings.pt'))


    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.learning_rate, weight_decay=Config.L2)
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot_e1 = torch.FloatTensor(Config.batch_size, num_entities)
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot_r = torch.FloatTensor(Config.batch_size, num_rel)

    model.eval()
    train_data =[]
    with open('data/'+Config.dataset+'/train.txt', 'r') as f:
        for i, line in enumerate(f):
            e1, rel, e2 = line.decode('utf-8').split('\t')
            e1 = e1.strip()#.lower()
            e2 = e2.strip()#.lower()
            rel = rel.strip()#.lower()
            train_data += [[e1, rel, e2]]
    print len(train_data)

    attack_list = []
    E2_list = []
    with open('data/'+Config.dataset+'/test.txt', 'r') as f:
        for i, line in enumerate(f):
            e1, rel, e2 = line.decode('utf-8').split('\t')
            e1 = e1.strip().lower()
            e2 = e2.strip().lower()
            rel = rel.strip().lower()
            attack_list += [[dict_tokentoid[e1], dict_reltoid[rel], dict_tokentoid[e2]]]
            E2_list += [e2]

    print len(attack_list)
    E2_list = set(E2_list)
    E2_dict = {}
    for i in train_data:
        if i[2].lower() in E2_list:
            if dict_tokentoid[i[2].lower()] in E2_dict: 
                E2_dict[dict_tokentoid[i[2].lower()]] += [[dict_tokentoid[i[0].lower()], dict_reltoid[i[1].lower()]]]
            else:
                E2_dict[dict_tokentoid[i[2].lower()]] = [[dict_tokentoid[i[0].lower()], dict_reltoid[i[1].lower()]]]


    str_at = []
    embd_e = model.emb_e.weight.data.cpu().numpy()
    embd_rel = model.emb_rel.weight.data.cpu().numpy()

    n_t = 0
    for trip in attack_list:
        if n_t % 500 == 0:
            print 'Number of processed triple: ', n_t

        n_t += 1
        e1 = trip[0]
        rel = trip[1]
        e2_or = trip[2]
        e1 = torch.cuda.LongTensor([e1])
        rel = torch.cuda.LongTensor([rel])
        e2 = torch.cuda.LongTensor([e2_or])
        pred = model.encoder(e1, rel)
        E2 = model.encoder_2(e2)
        
        A, B = find_best_at(-pred, E2)
        attack_ext = -A*pred+B*E2
        if e2_or in E2_dict:
            nei = E2_dict[e2_or]
            #attack = find_best_attack(E2.data.cpu().numpy(), pred.data.cpu().numpy(), nei, embd_e, embd_rel, attack_ext)
            #attack = torch.autograd.Variable(torch.from_numpy(attack)).cuda().float()
            attack = attack_ext

        else: 
            attack = attack_ext
        E1, R = model.decoder(attack)
        _, predicted_e1 = torch.max(E1, 1)
        _, predicted_R = torch.max(R, 1)

        str_at += [[str(dict_idtotoken[predicted_e1.data.cpu().numpy()[0]]), str(dict_idtorel[predicted_R.data.cpu().numpy()[0]]), str(dict_idtotoken[e2_or])]]


    new_train = str_at + train_data
    print len(new_train)
    with open('data/new_'+Config.dataset+'/train.txt', 'w') as f:
        for item in new_train:
            f.write("%s\n" % "\t".join(map(str, item)))


if __name__ == '__main__':
    main()
