import torch
import numpy as np
import datetime
import operator
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from spodernet.utils.global_config import Config
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.utils.logger import Logger
from torch.autograd import Variable
from sklearn import metrics

#timer = CUDATimer()
log = Logger('evaluation{0}.py.txt'.format(datetime.datetime.now()))

def ranking_and_hits(model, dev_rank_batcher, vocab, name, epoch, dict_idtotoken, dict_idtorel):
    log.info('')
    log.info('-'*50)
    log.info(name)
    log.info('-'*50)
    log.info('')
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    mrr_left = []
    mrr_right = []
    rel2ranks = {}


    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']
        e2_multi1 = str2var['e2_multi1'].float()
        e2_multi2 = str2var['e2_multi2'].float()
        pred1 = model.forward(e1, rel)
        pred2 = model.forward(e2, rel)
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
        for i in range(Config.batch_size):
            # these filters contain ALL labels
            filter1 = e2_multi1[i].long()
            filter2 = e2_multi2[i].long()

            # save the prediction that is relevant
            target_value1 = pred1[i,e2[i, 0]]
            target_value2 = pred2[i,e1[i, 0]]

            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2


        # sort and rank
        max_values1, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values2, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()

        for i in range(Config.batch_size):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i]==e2[i, 0])[0][0]
            rank2 = np.where(argsort2[i]==e1[i, 0])[0][0]

            ranks.append(rank1+1)
            ranks_left.append(rank1+1)
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

        dev_rank_batcher.state.loss = [0]  

    for i in range(10):
        log.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
        log.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
        log.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
    log.info('Mean rank left: {0}', np.mean(ranks_left))
    log.info('Mean rank right: {0}', np.mean(ranks_right))
    log.info('Mean rank: {0}', np.mean(ranks))
    log.info('Mean reciprocal rank left: {0}', np.mean(1./np.array(ranks_left)))
    log.info('Mean reciprocal rank right: {0}', np.mean(1./np.array(ranks_right)))
    log.info('Mean reciprocal rank: {0}', np.mean(1./np.array(ranks)))
    with open("Output.txt", "a") as text_file:
	text_file.write('{0}\n'.format(name))
        text_file.write('Hits @{0}: {1}\n'.format(1, np.mean(hits[0])))
	text_file.write('Hits @{0}: {1}\n'.format(3, np.mean(hits[2])))
	text_file.write('Hits @{0}: {1}\n'.format(10, np.mean(hits[9])))
	text_file.write('MRR: {0}\n'.format(np.mean(1./np.array(ranks))))
	text_file.write('epoch: {0}\n'.format(epoch))
	text_file.write('-------------------------------------------------\n')

def attack_tri(model, dev_rank_batcher, vocab, name, epoch, dict_idtotoken, dict_idtorel):
    log.info('')
    log.info('-'*50)
    log.info(name)
    log.info('-'*50)
    log.info('')
    attack_set = []
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    mrr_left = []
    mrr_right = []
    rel2ranks = {}
    dict_score = {}
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']
        e2_multi1 = str2var['e2_multi1'].float()
        e2_multi2 = str2var['e2_multi2'].float()
        pred1 = model.forward(e1, rel)
        pred2 = model.forward(e2, rel)
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
        for i in range(Config.batch_size):
            # these filters contain ALL labels
            filter1 = e2_multi1[i].long()
            filter2 = e2_multi2[i].long()

            # save the prediction that is relevant
            target_value1 = pred1[i,e2[i, 0]]
            target_value2 = pred2[i,e1[i, 0]]

            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2


        # sort and rank
        max_values1, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values2, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()

        
        for i in range(Config.batch_size):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i]==e2[i, 0])[0][0]
            rank2 = np.where(argsort2[i]==e1[i, 0])[0][0]

            if (rank1+rank2)/2.0 < 0.5:
                dict_score[(e1[i].cpu().numpy()[0], rel[i].cpu().data.numpy()[0], e2[i].cpu().numpy()[0])] = ((max_values1[i][rank1] - max_values1[i][rank1+1]) + (max_values2[i][rank2] - max_values2[i][rank2+1]))/2.0

            # rank+1, since the lowest rank is rank 1 not rank 0
        ############################### finding entity with hits @ 1
#       if dict_idtotoken[e1[i].cpu().numpy()[0]]=='judy_garland' and dict_idtorel[rel[i].cpu().data.numpy()[0]] =='ismarriedto':
#       print "judy:",rank1, rank2, target_value1, target_value2 
#       if dict_idtotoken[e1[i].cpu().numpy()[0]]=='manitoba' and dict_idtorel[rel[i].cpu().data.numpy()[0]] =='islocatedin':
#       print "manitoba:",rank1, rank2, target_value1, target_value2 
#       if dict_idtotoken[e1[i].cpu().numpy()[0]]=='sean_doherty_(footballer)' and dict_idtorel[rel[i].cpu().data.numpy()[0]] =='playsfor':
#       print "sean:",rank1, rank2, target_value1, target_value2 

            # if rank1<=2 and rank2<=2:
            #     attack_set +=  [[e1[i].cpu().numpy(), rel[i].cpu().data.numpy(), e2[i].cpu().numpy()]]
#       print dict_idtotoken[e1[i].cpu().numpy()[0]], dict_idtorel[rel[i].cpu().data.numpy()[0]], dict_idtotoken[e2[i].cpu().numpy()[0]]

    sorted_score = sorted(dict_score.items(), key=operator.itemgetter(1))
    # sorted_score = sorted_score[:100]
    # print len(sorted_score)
    # print sorted_score


    sorted_score = sorted_score[:100]
    for i in sorted_score:
        attack_set +=  [[dict_idtotoken[i[0][0]], dict_idtorel[i[0][1]], dict_idtotoken[i[0][2]]]]
        #attack_set +=  [(i[0][0], i[0][2])]

    print sorted_score
    print len(dict_score)
    with open('data/WN-18/new_test_4.txt', 'w') as f:
        for item in attack_set:
            #print item[0], item[1], item[2]
            f.write("%s\n" % "\t".join(map(str, item)))
    #np.save('data/attack_set', attack_set)
    print pouya
    return attack_set