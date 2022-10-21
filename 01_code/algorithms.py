import numpy as np
import matplotlib.pyplot as plt
import torch

from models import *

# this file contains algorithms to train cqrnn and baselines
# written as functions, returning models

def train_loop_single_models(model, x_train_torch, y_train_torch, cen_indicator_torch, global_kmf_torch, learning_rate, 
                            weight_decay, n_epochs, batch_size, loss_str, taus, taus_torch, n_data, n_quantiles,
                            IS_USE_CENSOR_LOSS, IS_USE_CROSS_LOSS, IS_FORCE_ALL_OBS, y_max, device, is_verbose):

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
    for ep in range(n_epochs):

        # reduce learning rate
        if ep == int(n_epochs*0.7):
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] /10
        if ep == int(n_epochs*0.9):
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] /10

        # do minibatches
        permutation = torch.randperm(x_train_torch.size()[0])
        loss_ep = 0.
        for i in range(0, x_train_torch.size()[0], batch_size):
            indices_batch = permutation[i:i+batch_size]
            x_train_batch, y_train_batch = x_train_torch[indices_batch], y_train_torch[indices_batch]
            cen_indicator_batch = cen_indicator_torch[indices_batch]
            global_kmf_torch_batch = global_kmf_torch[indices_batch]

            y_pred = model(x_train_batch.to(device))
            if loss_str in ['cqrnn','all_obs','excl_censor']: 
                loss = cqrnn_loss(y_pred, y_train_batch, cen_indicator_batch, taus, taus_torch, IS_USE_CENSOR_LOSS, IS_USE_CROSS_LOSS, IS_FORCE_ALL_OBS, y_max)
                # loss = cqrnn_loss_slowforloops(y_pred, y_train_batch, cen_indicator_batch, taus, IS_USE_CENSOR_LOSS, n_quantiles)
            elif loss_str == 'deepquantreg': 
                loss = deepquantreg_loss(y_pred, y_train_batch, cen_indicator_batch, global_kmf_torch_batch, taus, taus_torch, n_quantiles, IS_USE_CROSS_LOSS)
                # loss = deepquantreg_loss_slowforloops(y_pred, y_train_batch, cen_indicator_batch, global_kmf_torch_batch, taus, n_quantiles)
            elif loss_str == 'ucond_cqrnn':
                loss = neocleous_loss(y_pred, y_train_batch, cen_indicator_batch, taus, taus_torch, 0.999-global_kmf_torch_batch, IS_USE_CROSS_LOSS, y_max)
                # if weights=1 then get division by zero in loss, so smudge this a bit
            elif loss_str == 'lognorm': 
                loss = lognorm_loss(y_pred, y_train_batch, cen_indicator_batch)

            # gradient update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ep+=loss.detach().item()

        if is_verbose==1: print('epoch',ep,', loss',round(loss_ep/n_data,4),end='\r')

    return model


def train_loop_neocleous(models_neo, x_train_torch, y_train_torch, cen_indicator_torch, global_kmf_torch, learning_rate, 
                            weight_decay, n_epochs, batch_size, loss_str, taus, taus_torch, n_data, n_quantiles,
                            IS_USE_CENSOR_LOSS, IS_USE_CROSS_LOSS, IS_FORCE_ALL_OBS, y_max, device, is_verbose):
    # can also compare this to crq.fit.por2 in : https://github.com/cran/quantreg/blob/master/R/crq.R 
    # this is the newer version on 14 May

    for j in range(n_quantiles):
        if is_verbose==1: print('Neocleous grid training:', j+1 , ' of ', n_quantiles)
        optimizer = torch.optim.Adam(models_neo[j].parameters(),lr=learning_rate, weight_decay=weight_decay)

        # update estimated_quantiles
        # we do this for all datapoints, even though only need it for censored
        if j==0:
            # start w 1.0, it doesn't really matter what these are for censored loss
            # when the prediction is below that value
            # since the algorithm assumes all censored datapoints are below first quantile sought
            # it's a bit confusing in R code because for first grid point they init w's to 1, but subsequently they are init to 0
            # estimated_quantiles = torch.zeros(y_train_torch.shape[0],1) + taus[j] # this doesn't  impact loss for first epoch
            estimated_quantiles = torch.zeros(y_train_torch.shape[0],1)
            not_updated = estimated_quantiles>=0 # not updated
        else:
            # find out which data points are now crossed, which haven't been updated before
            y_preds = models_neo[j-1](x_train_torch).detach()
            larger_check = y_preds > y_train_torch
            estimated_quantiles[larger_check*not_updated] = taus[j-1]
            not_updated = torch.logical_and(not_updated, larger_check*not_updated==False)  # elements not updated are already not updated and were not updated just now
            estimated_quantiles[not_updated] = taus[j-1] # this is roughly equivalent to setting w=0, or can comment out to use q=0, both work well
            # estimated_quantiles[not_updated] = taus[j] 

        for ep in range(n_epochs):

            # reduce learning rate
            if ep == int(n_epochs*0.7):
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] /10
            if ep == int(n_epochs*0.9):
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] /10

            # do minibatches
            permutation = torch.randperm(x_train_torch.size()[0])
            loss_ep = 0.
            for i in range(0, x_train_torch.size()[0], batch_size):
                indices_batch = permutation[i:i+batch_size]
                x_train_batch, y_train_batch = x_train_torch[indices_batch], y_train_torch[indices_batch]
                cen_indicator_batch = cen_indicator_torch[indices_batch]
                estimated_quantiles_batch = estimated_quantiles[indices_batch]

                y_pred = models_neo[j](x_train_batch.to(device))
                loss = neocleous_loss(y_pred, y_train_batch, cen_indicator_batch, taus[j], taus_torch[:,j].view(1,1), estimated_quantiles_batch, False, y_max)

                # gradient update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_ep+=loss.detach().item()

            print('epoch',ep,', loss',round(loss_ep/n_data,4),end='\r')

    return models_neo
