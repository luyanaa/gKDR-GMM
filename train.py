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
    def __init__(self, input_size, lag, filter_size, num_components: int,
        X, y, mixture):
        super().__init__()
        self.ExpFilter = ExpFilter(input_size=input_size, lag=lag, filter_size=filter_size)
        self.shape = [X[i].shape[1] for i in range(len(X))]
        self.Yt = []
        cnt = 0
        for i in range(len(X)):
            self.Yt.append(cnt+X[i].shape[1]-2) # Yt-1
            self.Yt.append(cnt+X[i].shape[1]-1) # Yt
            cnt = cnt + X[i].shape[1]
        self.Yt = torch.LongTensor(self.Yt)
        if mixture == "full":
            self.GMM = [GmmFull(num_components=num_components, num_dims=self.shape[i]) for i in range(len(X))]
        elif mixture == "diagonal":
            self.GMM = [GmmDiagonal(num_components=num_components, num_dims=self.shape[i]) for i in range(len(X))]
        elif mixture == "isotropic":
            self.GMM = [GmmIsotropic(num_components=num_components, num_dims=self.shape[i]) for i in range(len(X))]
        elif mixture == "shared":
            self.GMM = [GmmSharedIsotropic(num_components=num_components, num_dims=self.shape[i]) for i in range(len(X))]
    def forward(self, X):
        # Remember shape of list X for GMM input.
        X = self.ExpFilter(X, self.Yt)
        cnt = 0
        y = []
        for i in range(len(self.shape)):
            nll_loss = self.GMM[i](X[:, cnt:cnt+self.shape[i]].squeeze(0).transpose(0,1))
            cnt = cnt+self.shape[i]
            y.append(nll_loss)
        return y

X=[]
Y=[]
def objective(trial):
    mixture = trial.suggest_categorical("mixture", ["diagonal", "full", "isotropic", "shared"])
    num_components = trial.suggest_int('num_components', 1, 5)
    filter_size = trial.suggest_int('filter_size', 10, 100, step=10)
    lag = trial.suggest_int('lag', 10, 100, step=10)
    gKDR_pivot = 20
    gKDR_List = []
    input_size = 0
    for i in range(len(X)):
        if X[i].shape[-1] <= gKDR_pivot:
            val = X[i][1:]
            val = torch.hstack((val, Y[i][:-1].unsqueeze(1), Y[i][1:].unsqueeze(1)))
            input_size = input_size + val.shape[1]
            gKDR_List.append(val)
            # Saving time, doing gKDR on too-low dimension is not reasonable. 
            continue

        K = trial.suggest_int('K_' + str(i), 1, X[i].shape[-1])
        # Currently only single C. elegans data, not dealing with batch.  
        if X[i].ndim == 2:
            val = gKDR(X[i][1:], Y[i][1:], K)(X[i][1:])
            # Adding Past Observation. 
            val = torch.hstack((val, Y[i][:-1].unsqueeze(1), Y[i][1:].unsqueeze(1)))
            input_size = input_size + val.shape[1]
            gKDR_List.append(val)
    mat = torch.zeros((gKDR_List[0].shape[0], input_size))
    cnt = 0
    for i in range(len(gKDR_List)):
        mat[:, cnt:cnt+gKDR_List[i].shape[1]] = gKDR_List[i]
        cnt = cnt + gKDR_List[i].shape[1]
    model = FilterGMM(input_size=input_size, lag = lag, filter_size=filter_size, num_components=num_components, X=gKDR_List, y=Y, mixture=mixture)
    mixture_lr = 0.05
    component_lr = 0.05
    num_iterations = 100
    log_freq = 5
    ExpFilter_lr = 0.05
    ExpFilter_Gamma = trial.suggest_float("ExpFilter_Gamma", 0.1, 0.5)
    # create separate optimizers for mixture coeficients and components
    ExpFilter_optimizer = torch.optim.Adam(model.ExpFilter.parameters(), lr=ExpFilter_lr)
    ExpFilter_scheduler = torch.optim.lr_scheduler.StepLR(ExpFilter_optimizer, step_size=num_iterations // 10, gamma=ExpFilter_Gamma)
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
            ExpFilter_optimizer.zero_grad()

        # forward
        output = model(mat)
        loss = 0
        for i in range(len(gKDR_List)):
            loss = loss + output[i] 
        trial.report(loss, iteration_index)
        if trial.should_prune():
            raise optuna.TrialPruned()

        loss.backward()

        # log and visualize
        # if log_freq is not None and iteration_index % log_freq == 0:
        #    print(f"Iteration: {iteration_index:2d}, Loss: {loss.item():.2f}")

        for i in range(len(gKDR_List)):
            mixture_optimizer[i].step()
            mixture_scheduler[i].step()
            components_optimizer[i].step()
            components_scheduler[i].step()
            ExpFilter_optimizer.step()
            ExpFilter_scheduler.step()
            model.GMM[i].constrain_parameters()

    return loss.detach()

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
        target_train = data[:,targeti]
        source_train = data[:,seli]
        Y.append(torch.from_numpy(target_train))
        X.append(torch.from_numpy(source_train))
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), 
                                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
                            )
    study.optimize(objective, n_trials=500)

# Redirect output back to stdout
sys.stdout = orig_stdout
f.close()


