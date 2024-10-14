import argparse
import json
import logging
import sys
from os import makedirs
from os.path import join as pjoin
from shutil import copy2, move

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.models.deep4 import Deep4Net
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds
from sklearn.model_selection import KFold
from SACNN import SACNN
import pickle




logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
parser = argparse.ArgumentParser(
    description='Subject-independent classification with Common Stroke Data')
parser.add_argument('-gpu', type=int, help='The gpu device to use', default=0)
args = parser.parse_args()

datapath = r'D:\MaJun\DataSet\SHU_Stroke_dataset.h5'
outpath = r'.\results'

PartialPrior=0.7    #   This parameter is  choosed from [0.1,0.3,0.5,0.7,0.9] in original paper.

subjs = list(np.arange(1, 23))
dfile = h5py.File(datapath, 'r')
torch.cuda.set_device(args.gpu)
set_random_seeds(seed=1, cuda=True)
BATCH_SIZE = 16
TRAIN_EPOCH = 200


def get_data(subj):
    dpath = r'/s' + str(subj)
    X = dfile[dpath + '/X']
    Y = dfile[dpath + '/Y']
    return X, Y.astype(np.int64)

def get_multi_data(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y

def GenertaMultiFolds(idx, kFold=10,trainratio=0.9):
    tlen=len(idx)
    trainNum=int(tlen*trainratio)
    step=(tlen-trainNum)/kFold
    trainfolds=[]
    for si in range(kFold):
        trainfolds.append(idx[int(si*step):trainNum+int(si*step)])
    return trainfolds


def reset_model(model,checkpoint):
    model.network.load_state_dict(checkpoint['model_state_dict'])
    for param in model.network.TemproalLayer.parameters():
        param.requires_grad = False
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.network.parameters()),
                      lr=0.0005, weight_decay=0.5*0.001)
    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1)

def TraingTest(SubIndex):

     # split dataset to source domain and target domain

    test_subj = subjs[SubIndex]
    ExceptTestSubjs = np.array(subjs[SubIndex+1:] + subjs[:SubIndex])
    train_subjs=ExceptTestSubjs[:int(len(ExceptTestSubjs)*0.9)]
    valid_subjs=ExceptTestSubjs[int(len(ExceptTestSubjs)*0.9):]
    S_X_train, S_Y_train = get_multi_data(train_subjs)
    S_X_val, S_Y_val = get_multi_data(valid_subjs)
    T_X_test_full, T_Y_test_full = get_data(test_subj)


    n_classes = 4
    in_chans = S_X_train.shape[1]   # channel
    in_time=S_X_train.shape[2]         # time samples


    fullIdxFold=np.arange(len(T_X_test_full))
    IdxFold = GenertaMultiFolds(fullIdxFold, 10)


    for j, cvfolds in enumerate(IdxFold):

        # train partial prior base model

        testIndex=list(fullIdxFold)
        [testIndex.remove(i) for i in cvfolds]
        T_X_test, T_Y_test = T_X_test_full[testIndex], T_Y_test_full[testIndex]
        T_X_train, T_Y_train = T_X_test_full[list(cvfolds)], T_Y_test_full[list(cvfolds)]
        S_pp_X_val=np.vstack((S_X_val,T_X_train[:int(len(T_X_train)*PartialPrior)]))
        S_pp_Y_val=np.hstack(((S_Y_val,T_Y_train[:int(len(T_X_train)*PartialPrior)])))
        Strain_set = SignalAndTarget(S_X_train, y=S_Y_train)
        Svalid_set = SignalAndTarget(S_pp_X_val, y=S_pp_Y_val)
        model = SACNN(in_chans=in_chans, n_classes=n_classes,
                      input_time_length=in_time).cuda()
        optimizer = AdamW(model.parameters(), lr=1 * 0.01, weight_decay=0.5 * 0.001)
        model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, )

        exp = model.fit(Strain_set.X, Strain_set.y, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, scheduler='cosine',
                        validation_data=(Svalid_set.X, Svalid_set.y), remember_best_column='valid_loss')
        rememberer = exp.rememberer
        base_model_param = {
            'epoch': rememberer.best_epoch,
            'model_state_dict': rememberer.model_state_dict,
            'optimizer_state_dict': rememberer.optimizer_state_dict,
            'loss': rememberer.lowest_val
        }
        torch.save(base_model_param, pjoin(
            outpath, 's{}_model_fold{}.pt'.format(test_subj,j)))
        model.epochs_df.to_csv(
            pjoin(outpath, 's{}_epochs_fold{}.csv'.format(test_subj,j)))

        #  split training set, validation set and test set in target domain

        valIndex=list(cvfolds[-1*int(len(fullIdxFold)*0.1):])
        T_X_val, T_Y_val = T_X_test_full[valIndex], T_Y_test_full[valIndex]
        trainIndex=list(cvfolds)
        [trainIndex.remove(i) for i in valIndex]
        T_X_train, T_Y_train = T_X_test_full[trainIndex], T_Y_test_full[trainIndex]

        # model transfer learning

        model = SACNN(in_chans=32, n_classes=4, input_time_length=1000).cuda()
        checkpoint = torch.load(pjoin(outpath, 's{}_model_fold{}.pt'.format(test_subj,j)), map_location='cuda:0')
        reset_model(model,checkpoint)
        model.fit(T_X_train, T_Y_train, epochs=TRAIN_EPOCH,
                  batch_size=BATCH_SIZE, scheduler='cosine',
                  validation_data=(T_X_val, T_Y_val), remember_best_column='valid_loss')
        model.epochs_df.to_csv(pjoin(outpath, 's' + str(test_subj) +'_fold_'+str(j)+ '.csv'))

        # model test

        test_loss = model.evaluate(T_X_test, T_Y_test)
        with open(pjoin(outpath, 'test_s' + str(test_subj) +'_fold_'+str(j)+ '.json'), 'w') as f:
            json.dump(test_loss, f)
        accuracy=1-len(np.nonzero(model.predict_classes(T_X_test)-T_Y_test)[0]) / len(model.predict_classes(T_X_test)-T_Y_test)
        res=[test_subj,j,accuracy,model.predict_outs(T_X_test),model.predict_classes(T_X_test),T_Y_test]
        with open(pjoin(outpath,'s_{}_fold{}_results.pkl'.format(test_subj,j)),'wb') as f:
            pickle.dump(res,f)

    dfile.close()

if __name__=='__main__':
    TraingTest(0)   # Subj index from 0 to 21, 22 subjects in all.
