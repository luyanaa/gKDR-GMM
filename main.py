# Translated by Robin Lu to Python, with certain change/improvement. 
# Should not be taken as equvilant of original gKDR-GMM paper. 
import scipy.signal
import numpy
import pandas as pd
import os
import argparse
import gKDR
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

def gmm_bic_score(estimator, X, y):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -mean_absolute_error(y, estimator.pred(X))
def unique(seq): # Order preserving
  ''' Modified version of Dave Kirby solution '''
  seen = set()
  return [x for x in seq if x not in seen and not seen.add(x)]
# Requirements:
# [common_data_folder]/samplex_data.mat
# [metadata_folder]/conneurons.csv, multiconmatrix.csv
# where x stands for sample number.

# sampleID: sample number (command line input = UGE_TASK_ID_text)
# cellc: cell number in conNames (connection table)
# cellt: cell number in targetcells
# note that connectome data are given based on conNames
# selc: cells within link from target cell, number in conNames
# seli: cells within link from target cell, number in uniqNames

parser = argparse.ArgumentParser(
                    prog='gKDR-GMM Python version',
                    description='An improvement of Toyoshima et al. (2023)',
                    epilog='Under Development')

sampleID = 1
data_folder = "./cleandata_smoothened2"
metadata_folder = "./metadata" 
# Preset Parameters

link = 'indirect'

embed_width = 30 # number of embed_step's used for embedding = column number in source data
embed_step = 10 # invervals used for embedding (index-based)
time_step = 5 # timestep invervals in which estimation is performed
nahead = 5 # how far ahead is the target for 

show_plots = False
use_salt_input = False; # modified by YT
saltsensors = {'ASEL','ASER','BAGL','BAGR','AWCL','AWCR','ASHL','ASHR'}

# Initial treatment
conNames = pd.read_csv(os.path.join(metadata_folder, 'conneurons.csv'),header=None)

autocorrthreshold = 0.3
autocorrlag = 20
data = pd.read_csv(os.path.join(data_folder, str(sampleID)+'_ratio.csv'));
uniqNames = pd.read_csv(os.path.join(data_folder, str(sampleID)+'_uniqNames.csv'),header=None)    
rall = numpy.diag(numpy.corrcoef(data[1+autocorrlag:-1],data[1:-1-autocorrlag]))
targetcells = numpy.where(rall > autocorrthreshold)
targetcellnames = uniqNames[targetcells]
targetcells = []
for i in range(targetcellnames):
    if numpy.any(conNames==targetcellnames[i]):
        targetcells.append(numpy.where(uniqNames==targetcellnames[i])) 
    
Mt = len(targetcells)
targetcellnames = uniqNames(targetcells)
print('start')
print('link type = ', link)
print('<< sample ', str(sampleID), ' >>')

selistart = 1 - use_salt_input
embed_size = embed_width

# read connection data
multiconmatrix = pd.read_csv(os.path.join(metadata_folder, 'multiconmatrix.csv'),header=None)

n = data.shape[0]
print("n is ", n)
train_span = [1, n]
test_span = []
        
def generatesalt(n, startframe, period):
    x = numpy.arange(1, n+1, 1) - startframe
    y = numpy.dot(numpy.math.epower(numpy.abs(numpy.sin(x/period*numpy.pi)), 0.25), numpy.sign(numpy.sin(x/period*numpy.pi)))
    y[1:numpy.floor(startframe)] = 0
    return y

if use_salt_input:
    salttable = pd.read_csv(os.path.join(metadata_folder, 'stimulation_timing.csv'),header=1) 
    startframe = salttable[sampleID,3]
    period = salttable[sampleID,4];
    saltdata = generatesalt(n, startframe, period);

embed_before = embed_width-1
  
# embed as a bulk
print('execute embedding')
if use_salt_input:
    sourcedata = [saltdata[1:n].transpose(0,1), data[:,targetcells]]
else:
    sourcedata = data[:,targetcells]
# print(sourcedata.shape)
# [source_train, target_train, source_test, target_test] = ...
def embed4D(source_data, target_data, train_span, test_span, embed_step, embed_before, time_step, nahead):
    embed_size = embed_before + 1
    M = source_data.shape[1] # number of time series used for source signal
    # set embedded data: source_train, target_train
    # columns for source_train: data1(t-embed_before*embed_step),,,,,, data1(t),data2(t-embed_before*embed_step),,,,,, data2(t),
    start_ind = train_span[0] + embed_before * embed_step
    end_ind = train_span[1] - nahead
    ind_seq = numpy.arange(start=start_ind, stop=end_ind+time_step, step=time_step) # sequence of time indices to pick up for training
    source_train = numpy.zeros((ind_seq.shape[0], embed_size*M))
    source_train[:,:] = numpy.nan
    columns = numpy.arange(start=1, step=embed_size, stop=(1+embed_size*(M-1))+embed_size)
    for shift in range((-embed_before)*embed_step, embed_step, embed_step):
        print('shift = ', str(shift))
        source_train[:, columns] = source_data[ind_seq + shift, :]   # from old to now
        columns = columns + 1 # add to each column number
    target_train = target_data[ind_seq+nahead,:]
    if test_span.shape[0] < 2:
        source_test = []
        target_test = []
    else:
        start_ind = test_span[0] + embed_before * embed_step
        end_ind = test_span[1] - nahead;
        print(start_ind, end_ind)

        ind_seq = numpy.arange(start=start_ind,stop=end_ind+time_step, step=time_step) # sequence of time indices to pick up
        source_test = numpy.zeros((ind_seq.shape[0], embed_size*M))
        source_test[:,:] = numpy.nan
        columns = numpy.arange(start=1, step=embed_size, stop=(1+embed_size*(M)))
        # embed
        for shift in range((-embed_before*embed_step), embed_step , embed_step):
            print('shift = ', str(shift))
            source_test[:, columns] = source_data[ind_seq + shift, :]
            columns = columns + 1
        target_test = target_data[ind_seq+nahead,:]
    return source_train, target_train, source_test, target_test

source_train_all, target_train_all, _, _ = embed4D(sourcedata, data[:,targetcells], train_span, test_span, embed_step, embed_before, time_step, nahead)
print('end of embedding')

model = dict()
# target cell loop
for targeti in range(targetcells):
    targetcellname = targetcellnames[targeti]
    cellc = pd.where(conNames == targetcellname )
    if cellc.size ==0:
        print('There is no cell in connection table named ',targetcellname)
        continue
    # find connected neurons
    if link == 'direct':
        selc = numpy.where(multiconmatrix[:,cellc] == 1) # in numbers in conNames
        selc = list(selc)
    elif link == 'indirect':
        selc = numpy.where(multiconmatrix[:,cellc] == 1)
        selcall = selc
        for selci in list(selc):
            if numpy.where(targetcellnames == conNames[selci]).size() != 0 : # pre��targetcell�ɂȂ�������P�X�e�b�v����g��
                selcall.append(numpy.where(multiconmatrix[:,selci] == 1))
        selc = unique(selcall)
    elif link == 'all':
        selc = list(numpy.linspace(1, len(conNames)))
    
    selc[conNames[selc]==targetcellname] = []  
    selc = cellc + selc 
    connectedcellnames = conNames(selc)
    seli = []  # numbers in targetcells
    for i in range(len(connectedcellnames)):
        seli.append(numpy.where(uniqNames[targetcells]==connectedcellnames[i])) 
    sourcecellnames = targetcellnames[seli]
                    
    if use_salt_input and numpy.any(saltsensors == targetcellname):
        seli = [seli[0],0,seli[1:-1]]
        sourcecellnames = [sourcecellnames[0], 'salt', sourcecellnames[1:-1]]
    
    print('sample ', str(sampleID), ' target ',targetcellname, ', No of linked neurons = ', str(len(selc)-1) )
    print('number of presynaptic on connectome: ', str(numpy.sum(multiconmatrix[:,cellc] == 1)-1));
    print('No of linked and annotated neurons = ', str(len(seli)-1), '( ',  targetcellnames[seli[seli>0]], ')') # uniqNames{targetcells(seli)}) ')']);
    target_train = target_train_all[:,targeti]
                    
    cols = numpy.reshape(numpy.repeat((1-selistart+seli-1)*numpy.ones(embed_size), embed_size, 1)+numpy.repeat(numpy.linspace(1,embed_size,embed_size).transpose(),1,len(seli)),[-1,1]).transpose()
                    
    source_train = source_train_all[:, cols]                   
    
    # Selecting Hyperparameters
    param_grid = {
        "gKDR__Kexe" : range(3, min( source_train.shape[1],max(cols.shapes))), # change K depending on the size of cols
        "gKDR__sgx": gKDR.MedianDist(source_train)*range(0, 2, 0.05), 
        "gKDR__sgy": gKDR.MedianDist(target_train)*range(0, 2, 0.05), 
        "gKDR__eps": 0.00001, 
        "GMM__n_components": range(1, 2),
        "GMM__covariance_type": ["spherical", "tied", "diag", "full"],
    }
    class gKDR_Wrapper: 
        def __init__(self, Kexe, sgx, sgy, eps): 
            self.Kexe = Kexe
            self.sgx = sgx
            self.sgy = sgy
            self.eps = eps
        def fit(self, X, y):
            self.B,_,_,_=gKDR.KernelDeriv_chol(X, y, self.Kexe, self.sgx, self.sgy, self.eps)
        def transform(self, X, y):
            return X*self.B, y

    # gKDR & GMM
    grid_search = GridSearchCV(
        Pipeline([('gKDR', gKDR_Wrapper()), ('GMM', GaussianMixture())]), param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(source_train, y=target_train)
    model[targeti] = grid_search

# Freerun test.
# freerun_length = 10000 # number of time_step's to perform freerun prediction
# freerun_repeat = 3

#freerun_start = train_span[1]+1
# freestart = freerun_start
#if freestart > n+1:
#    freestart = n+1
        

#Tc_real = numpy.arange(train_span[0], (freestart-1)+time_step, time_step) # sequence of time indices for real data before freerun
#Tc_all = numpy.arange(train_span[0],(freestart+freerun_length)+time_step, time_step)
#nx = Tc_all.shape[0]    # Tc_all from saveddata  XXXXXXXXXXX
#nc = Tc_real.shape[0]    # Tc_real from saveddata  XXXXXXXXXXX
#print('Freerun prediction')
#embed_time_step = embed_step/time_step # embed_stepがtime_step単位でいくつか
#ahead_time_step = nahead/time_step # naheadがtime_step単位でいくつか
        
#for testi in range(freerun_repeat):        
#    print('repeat ', str(testi))
#    real_predict = numpy.nan((len(Tc_all),len(targetcells)+1-selistart)) # add if salt input            
    # put real data at the begining of real_predict
#    real_predict[1:len(Tc_real), 1-selistart+(1:len(targetcells))] = data[Tc_real, targetcells]
#    if use_salt_input:
#        real_predict(1:len(Tc_all),1) = saltdata(Tc_all); # salt input      %%% saltdata from modeldata   
        # freerun prediction
#        for ti in range(1,(nx-nc)+1):
#            for targeti in range(len(targetcells)):
#                source_for_predict = numpy.reshape(real_predict(numpy.arange(start=(nc+ti-ahead_time_step-embed_time_step*embed_before),step=embed_time_step, stop=(nc+ti-ahead_time_step)+embed_time_step), selicell{targeti}+1-selistart),[],1).transpose() #  nc+tiをpredictしたいので、そこからnaheadもどったところがembedの終点になる
#                predict_ahead = GMMpredict(source_for_predict, model[targeti])
#                real_predict[nc+ti, targeti+1-selistart] = predict_ahead