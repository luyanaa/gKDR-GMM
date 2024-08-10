import optuna, torch, os, numpy
import pandas as pd
from ExpFilter import ExpFilter
from torch import nn

# Local Libraries
from gKDR import gKDR

from GMMPytorch import GmmDiagonal, GmmFull, GmmIsotropic, GmmSharedIsotropic
import gpytorch

# Logging stdout to log
import sys
orig_stdout = sys.stdout
f = open('out.log', 'w')
sys.stdout = f

def unique(seq): # Order preserving
  ''' Modified version of Dave Kirby solution '''
  seen = set()
  return [x for x in seq if x not in seen and not seen.add(x)]

class FilterGMM(nn.Module):
    def __init__(self, input_size, num_components: int,
        X, mixture):
        super().__init__()
        if mixture == "full":
            self.GMM = [GmmFull(num_components=num_components, num_dims=X[i].shape) for i in range(len(X))]
        elif mixture == "diagonal":
            self.GMM = [GmmDiagonal(num_components=num_components, num_dims=X[i].shape) for i in range(len(X))]
        elif mixture == "isotropic":
            self.GMM = [GmmIsotropic(num_components=num_components, num_dims=X[i].shape) for i in range(len(X))]
        elif mixture == "shared":
            self.GMM = [GmmSharedIsotropic(num_components=num_components, num_dims=X[i].shape) for i in range(len(X))]
    def forward(self, X):
        # Feed input of GMM one-by-one.
        y = []
        for i in range(len(X)):
            mixture_model = self.GMM[i](X[i].transpose(0,1))
            nll_loss = -1 * mixture_model.log_prob(X[i].transpose(0,1))
            y.append(nll_loss)
        return y

X=[]
Y=[]
def objective(trial):
    mixture = trial.suggest_categorical("mixture", ["diagonal", "full", "isotropic", "shared"])
    num_components = trial.suggest_int('num_components', 1, 5)
    gKDR_pivot = 20
    gKDR_List = []
    input_size = 0
    for i in range(len(X)):
        reducion = gKDR(X[i][1:], Y[i][1:], K)
        val = reduction(X[i][1:])
        # Adding Past Observation.
        val = torch.hstack((val, Y[i][:-1].unsqueeze(1), Y[i][1:].unsqueeze(1)))
        input_size = input_size + val.shape[1]
        gKDR_List.append(val)
        if free_run:
            val_test = reduction(X_test[i][1:])
            val_test = torch.hstack((val_test, Y_test[i][:-1].unsqueeze(1), Y_test[i][1:].unsqueeze(1)))
                input_size_test = input_size_test + val_test.shape[1]
                gKDR_List_test.append(val_test)
    model = FilterGMM(num_components=num_components, X=gKDR_List, mixture=mixture)
    mixture_lr = 0.05
    component_lr = 0.05
    num_iterations = 100
    log_freq = 5
    # create separate optimizers for mixture coeficients and components
    mixture_optimizer = [torch.optim.Adam(model.GMM[i].mixture_parameters(), lr=mixture_lr) for i in range(len(gKDR_List))]
    mixture_scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(mixture_optimizer[i], num_iterations) for i in range(len(gKDR_List))]
    components_optimizer = [torch.optim.Adam(model.GMM[i].component_parameters(), lr=component_lr) for i in range(len(gKDR_List))]
    components_scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(components_optimizer[i], num_iterations) for i in range(len(gKDR_List))]

    # optimize
    for iteration_index in range(num_iterations):
        # reset gradient
        for i in range(len(gKDR_List)):
            mixture_optimizer[i].zero_grad()
            components_optimizer[i].zero_grad()

        # forward
        output = model(gKDR_List)
        loss = 0
        for i in range(len(gKDR_List)):
            loss = loss + output[i] 
        trial.report(loss, iteration_index)
        if trial.should_prune():
            raise optuna.TrialPruned()
        loss.backward()

        for i in range(len(gKDR_List)):
            mixture_optimizer[i].step()
            mixture_scheduler[i].step()
            components_optimizer[i].step()
            components_scheduler[i].step()
            model.GMM[i].constrain_parameters()
    if free_run:
        # Testing Mode
        with torch.no_grad():
            # We still needs to filter the signal in inference mode. 
            cnt = 0
            y = []
            for i in range(len(self.shape)):
                mixture_model = model[i]
                # Get mean log probability for each neuron over time. 
                log_prob = mixture_model.log_prob(X[i].transpose(0,1)).mean()
                log_prob_GMM.append(log_prob)
        # Print log_prob for each neuron. 
        print(log_prob_GMM)
            
    return loss.detach()
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

def generatesalt(n, startframe, period):
    x = numpy.arange(1, n+1, 1) - startframe
    y = numpy.dot(numpy.math.epower(numpy.abs(numpy.sin(x/period*numpy.pi)), 0.25), numpy.sign(numpy.sin(x/period*numpy.pi)))
    y[1:numpy.floor(startframe)] = 0
    return y

# Data Input
data_folder = "./cleandata_smoothened2"
metadata_folder = "./metadata"
conNames = pd.read_csv(os.path.join(metadata_folder, 'conneurons.csv'),header=None).to_numpy()
multiconmatrix = pd.read_csv(os.path.join(metadata_folder, 'multiconmatrix.csv'),header=None).to_numpy()

# Preset Parameters

link = 'indirect'

time_step = 5 # timestep invervals in which estimation is performed
nahead = 5 # how far ahead is the target for 

show_plots = False
use_salt_input = False; # modified by YT
selistart = 1 - use_salt_input
saltsensors = {'ASEL','ASER','BAGL','BAGR','AWCL','AWCR','ASHL','ASHR'}

autocorrthreshold = 0.3
autocorrlag = 20

print('link type = ', link)
for sampleID in range(1, 2):
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

    n = data.shape[0]
    train_span = numpy.array([1, n])
    test_span = numpy.array([])
    if use_salt_input:
        salttable = pd.read_csv(os.path.join(metadata_folder, 'stimulation_timing.csv'),header=1)
        startframe = salttable[sampleID,3]
        period = salttable[sampleID,4];
        saltdata = generatesalt(n, startframe, period);
    if use_salt_input:
        sourcedata = [saltdata[1:n].transpose(0,1), data[:,targetcells]]
    else:
        sourcedata = data[:,targetcells]
    source_train_all, target_train_all, _, _ = embed4D(sourcedata, data[:,targetcells], train_span, test_span, embed_step, embed_before, time_step, nahead)
    for targeti in range(targetcells.shape[0]):
        targetcellname = targetcellnames[targeti]
        cellc = numpy.where(conNames == targetcellname )[0]
        if cellc.size ==0:
            print('There is no cell in connection table named ',targetcellname)
            continue
        # find connected neurons
        if link == 'direct':
            selc = numpy.where(multiconmatrix[:,cellc] == 1)[0] # in numbers in conNames
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
        selc = numpy.delete(selc, numpy.where(conNames[selc]==targetcellname)[0])
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
        target_train = target_train_all[:,targeti]
        cols = numpy.reshape(numpy.repeat((1-selistart+seli-1)*embed_size, 
                                          repeats=embed_size, axis=0) + numpy.repeat(numpy.arange(start=1,stop=embed_size+1,step=1),
                                                                                     repeats=len(seli), axis=0),[-1,1]).transpose()
        source_train = source_train_all[:, cols]
        source_train = source_train.squeeze(1)
        target_train = numpy.expand_dims(target_train, 1)
        Y.append(torch.from_numpy(target_train))
        X.append(torch.from_numpy(source_train))
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), 
                                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
                            )
    study.optimize(objective, n_trials=500)
    objective(study.best_trial)

# Redirect output back to stdout
sys.stdout = orig_stdout
f.close()


