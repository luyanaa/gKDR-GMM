import optuna

from ExpFilter import ExpFilter
from torch import nn
from gKDR import gKDR
import torch
from GMMPytorch import GmmDiagonal, GmmFull, GmmIsotropic, GmmSharedIsotropic

class FilterGMM(nn.Module):
    def __init__(self, input_size, lag, filter_size, num_components: int,
        num_dims: int, mixture, neuron_size):
        self.ExpFilter = ExpFilter(input_size=input_size, lag=lag, filter_size=filter_size)
        if mixture == "diagonal":
            self.GMM = nn.ModuleList([GmmDiagonal(num_components=num_components, num_dims=num_dims) for _ in range(neuron_size)])
        elif mixture == "full":
            self.GMM = nn.ModuleList([GmmFull(num_components=num_components, num_dims=num_dims) for _ in range(neuron_size)])
        elif mixture == "isotropic":
            self.GMM = nn.ModuleList([GmmIsotropic(num_components=num_components, num_dims=num_dims) for _ in range(neuron_size)])
        elif mixture == "shared":
            self.GMM = nn.ModuleList([GmmSharedIsotropic(num_components=num_components, num_dims=num_dims) for _ in range(neuron_size)])
    def forward(self, x):
        # Remember shape of list X for GMM input. 
        size = []
        _x = []
        if x[0].shape == 1 or x[0].shape == 2:
            for i in range(len(x)): 
                size.append(x[i].shape)
                _x = torch.stack(_x, x[i])
        _x = self.ExpFilter(_x)
        cnt = 0
        x_ = []
        for i in range(len(size)):
            x_.append(_x[cnt:cnt+size[i][-1]])
        y = self.GMM(x_)
        return y

global neuron_size
global X, Y
def objective(trial):
    mixture = trial.suggest_categorical("mixture", ["diagonal", "full", "isotropic", "shared"])
    num_components = trial.suggest_int('num_components', 1, 5)
    filter_size = trial.suggest_int('filter_size', 10, 100, step=10)
    lag = trial.suggest_int('lag', 10, 100, step=10)
    gKDR_List = []
    input_size = 0
    for i in range(neuron_size): 
        K = trial.suggest_int('K_' + str(neuron_size), 1, X[i].shape[-2])
        if X.shape == 2:
            val = gKDR(X[i], Y[i], K)(X[i])
            input_size = input_size + val.shape[0]

        else: 
            val = [gKDR(X[n][i], Y[n][i], K)(X[n][i]) for n in X.shape[0]]
            input_size = input_size + val[0].shape[0]
        gKDR_List.append(val)

    model = FilterGMM(input_size=input_size, lag = lag, filter_size=filter_size, num_components=num_components, mixture=mixture)
    mixture_lr = 0.05
    component_lr = 0.05
    num_iterations = 100
    log_freq = 5
    criterion = nn.functional.mse_loss
    mixture_optimizer = torch.optim.Adam(model.GMM.mixture_parameters(), lr=mixture_lr)
    mixture_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mixture_optimizer, num_iterations)
    components_optimizer = torch.optim.Adam(model.GMM.component_parameters(), lr=component_lr)
    components_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(components_optimizer, num_iterations)

    # optimize
    for iteration_index in range(num_iterations):
        # reset gradient
        components_optimizer.zero_grad()
        mixture_optimizer.zero_grad()

        # forward
        loss = criterion(model(gKDR_List), Y)
        print(loss)

        # log and visualize
        if log_freq is not None and iteration_index % log_freq == 0:
            print(f"Iteration: {iteration_index:2d}, Loss: {loss.item():.2f}")

        # backwards
        loss.backward()
        mixture_optimizer.step()
        mixture_scheduler.step()
        components_optimizer.step()
        components_scheduler.step()

        # constrain parameters
        model.GMM.constrain_parameters()

    return loss.detach()


study = optuna.create_study(sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=300)
