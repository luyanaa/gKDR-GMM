from typing import Optional

import torch

# Reference fitter, the model here is able to auto-diffrentation. 
def fit_model(
    model: torch.nn.Module,
    data: torch.Tensor,
    num_iterations: int,
    mixture_lr: float,
    component_lr: float,
    log_freq: Optional[int] = None,
    visualize: bool = True
) -> float:
    # create separate optimizers for mixture coeficients and components
    mixture_optimizer = torch.optim.Adam(model.mixture_parameters(), lr=mixture_lr)
    mixture_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mixture_optimizer, num_iterations)
    components_optimizer = torch.optim.Adam(model.component_parameters(), lr=component_lr)
    components_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(components_optimizer, num_iterations)

    # optimize
    for iteration_index in range(num_iterations):
        # reset gradient
        components_optimizer.zero_grad()
        mixture_optimizer.zero_grad()

        # forward
        loss = model(data)
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
        model.constrain_parameters()

    return float(loss.detach())