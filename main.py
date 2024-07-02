# Translated by Robin Lu to Python, with certain change/improvement. 
# Should not be taken as equvilant of original gKDR-GMM paper. 
import scipy.signal
import numpy
import pandas as pd
import os
import argparse
import gKDR_MOGP
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
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
conNames = pd.read_csv(os.path.join(metadata_folder, 'conneurons.csv'),header=None).to_numpy()

autocorrthreshold = 0.3
autocorrlag = 20
data = pd.read_csv(os.path.join(data_folder, str(sampleID)+'_ratio.csv')).to_numpy()
uniqNames = pd.read_csv(os.path.join(data_folder, str(sampleID)+'_uniqNames.csv'),header=None).to_numpy()

A = data[autocorrlag:]
B = data[:-autocorrlag]
Corr = numpy.absolute(numpy.corrcoef(A.T, B.T))
Corr = Corr[0:A.shape[1],-B.shape[1]:]
rall = numpy.diag(Corr)
targetcells = numpy.where(rall > autocorrthreshold)[0]
targetcellnames = list(uniqNames[targetcells])
targetcells = []
for i in range(len(targetcellnames)):
    if numpy.any(conNames==targetcellnames[i]):
        targetcells.append(numpy.where(uniqNames==targetcellnames[i])[0])
Mt = len(targetcells)
targetcells = numpy.array(targetcells).flatten()
targetcellnames = uniqNames[targetcells]
print('start')
print('link type = ', link)
print('<< sample ', str(sampleID), ' >>')

selistart = 1 - use_salt_input
embed_size = embed_width

# read connection data
multiconmatrix = pd.read_csv(os.path.join(metadata_folder, 'multiconmatrix.csv'),header=None).to_numpy()

n = data.shape[0]
print("n is ", n)
train_span = numpy.array([1, n])
test_span = numpy.array([])
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

def embed4D(source_data, target_data, train_span, test_span, embed_step, embed_before, time_step, nahead):
    embed_size = embed_before + 1
    M = source_data.shape[1] # number of time series used for source signal
    # set embedded data: source_train, target_train
    # columns for source_train: data1(t-embed_before*embed_step),,,,,, data1(t),data2(t-embed_before*embed_step),,,,,, data2(t),
    start_ind = train_span[0] + embed_before * embed_step
    end_ind = train_span[1] - nahead
    ind_seq = numpy.arange(start=start_ind, stop=end_ind, step=time_step) # sequence of time indices to pick up for training
    source_train = numpy.zeros((ind_seq.shape[0], embed_size*M))
    source_train[:,:] = numpy.nan
    columns = numpy.arange(start=0, step=embed_size, stop=(1+embed_size*(M-1)))
    for shift in range((-embed_before)*embed_step, embed_step, embed_step):
        source_train[:, columns] = source_data[ind_seq + shift, :]   # from old to new
        columns = columns + 1 # add to each column number
    target_train = target_data[ind_seq+nahead,:]
    if test_span.shape[0] < 2:
        source_test = []
        target_test = []
    else:
        start_ind = test_span[0] + embed_before * embed_step
        end_ind = test_span[1] - nahead;
        ind_seq = numpy.arange(start=start_ind,stop=end_ind+time_step, step=time_step) # sequence of time indices to pick up
        source_test = numpy.zeros((ind_seq.shape[0], embed_size*M))
        source_test[:,:] = numpy.nan
        columns = numpy.arange(start=1, step=embed_size, stop=(1+embed_size*(M)))
        # embed
        for shift in range((-embed_before*embed_step), embed_step , embed_step):
            source_test[:, columns] = source_data[ind_seq + shift, :]
            columns = columns + 1
        target_test = target_data[ind_seq+nahead,:]
    return source_train, target_train, source_test, target_test

source_train_all, target_train_all, _, _ = embed4D(sourcedata, data[:,targetcells], train_span, test_span, embed_step, embed_before, time_step, nahead)
print('end of embedding')

model = dict()
# target cell loop
for targeti in range(targetcells.shape[0]):
    targetcellname = targetcellnames[targeti]
    cellc = numpy.where(conNames == targetcellname )[0]
    if cellc.size ==0:
        print('There is no cell in connection table named ',targetcellname)
        continue
    # find connected neurons
    if link == 'direct':
        selc = numpy.where(multiconmatrix[:,cellc] == 1) # in numbers in conNames
    elif link == 'indirect':
        selc = numpy.where(multiconmatrix[:,cellc] == 1)[0]
        selcall = list(selc)
        for selci in list(selc):
            if numpy.where(targetcellnames == conNames[selci])[0].size != 0 : 
                selcall.append(numpy.where(multiconmatrix[:,selci] == 1))
        selcall = numpy.array(selc)
        selc = unique(selcall)
    elif link == 'all':
        selc = numpy.linspace(1, len(conNames))
    
    # selc[numpy.where(conNames[selc]==targetcellname)[0]] = []  
    selc = numpy.delete(selc, numpy.where(conNames[selc]==targetcellname)[0])
    # selc = cellc + selc
    selc = numpy.array(list(cellc)+list(selc))
    connectedcellnames = conNames[selc]
    seli = []  # numbers in targetcells
    for i in range(len(connectedcellnames)):
        result = numpy.where(uniqNames[targetcells]==connectedcellnames[i])[0]
        if result.size == 0: 
            continue
        seli.append(result)
    sourcecellnames = targetcellnames[seli]
    seli = numpy.array(seli)  
    if use_salt_input and numpy.any(saltsensors == targetcellname):
        seli = [seli[0],0,seli[1:-1]]
        sourcecellnames = [sourcecellnames[0], 'salt', sourcecellnames[1:-1]]
        seli = numpy.array(seli)
    seli = seli.flatten()
    print("seli.shape", seli.shape)
    print('sample ', str(sampleID), ' target ',targetcellname, ', No of linked neurons = ', str(len(selc)-1) )
    print('number of presynaptic on connectome: ', str(numpy.sum(multiconmatrix[:,cellc] == 1)-1));
    print('No of linked and annotated neurons = ', str(len(seli)-1), '( ',  targetcellnames[seli[seli>0]], ')') # uniqNames{targetcells(seli)}) ')']);
    target_train = target_train_all[:,targeti]
    cols = numpy.reshape(numpy.repeat((1-selistart+seli-1)*embed_size, repeats=embed_size, axis=0)+numpy.repeat(numpy.arange(start=1,stop=embed_size+1,step=1), repeats=len(seli), axis=0),[-1,1]).transpose()
    source_train = source_train_all[:, cols]
    source_train = source_train.squeeze(1)
    target_train = numpy.expand_dims(target_train, 1)
    # Selecting Hyperparameters
    param_grid = {
        "Kexe" : range(3, min( source_train.shape[1],max(cols.shape)), 9), # change K depending on the size of cols
    }
    print("source_train", source_train.shape, "target_train", target_train.shape)
    class gKDR_Wrapper(BaseEstimator): 
        def __init__(self, Kexe=None): 
            self.Kexe = Kexe
        def fit(self, X, y=None):
            X = source_train
            y = target_train
            self.gKDR=gKDR_MOGP.gKDR(X=numpy.hstack((X[1:], y[:-1])), Y=y[1:], K=self.Kexe)
            return self
        def transform(self, X, y=None):
            X = self.gKDR(X)
            return X
        def score(self, X, y):
            return self.gKDR._compute_loss(numpy.hstack((X[1:], y[:-1])), y[1:], 10)
    # gKDR & GMM
    grid_search = GridSearchCV(
        gKDR_Wrapper(), param_grid=param_grid, error_score="raise"
    )

    grid_search.fit(source_train, y=target_train)
    model[targeti] = grid_search

