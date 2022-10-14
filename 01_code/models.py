import torch
import torch.nn as nn
from scipy.stats import lognorm
import numpy as np

# this file contains model definitions and losses

class Model_linear(nn.Module):
	# linear model
	def __init__(self,input_dim,n_hidden,n_quantiles):
		super(Model_linear,self).__init__()
		self.layer = nn.Linear(input_dim,n_quantiles,bias=True) # in_features, out_features, bias

	def forward(self,x):
		x = self.layer(x)
		return x

class Model_mlp(nn.Module):
	# NN with two relu hidden layers
	# quantile outputs are independent of eachother
	def __init__(self,input_dim,n_hidden,n_quantiles,is_dropout=False,is_batch=False,activation='relu'):
		super(Model_mlp,self).__init__()
		self.layer1 = nn.Linear(input_dim,n_hidden,bias=True)
		self.layer2 = nn.Linear(n_hidden,n_hidden,bias=True)
		self.layer3 = nn.Linear(n_hidden,n_quantiles,bias=True)
		self.drop1 = nn.Dropout(0.333)
		self.drop2 = nn.Dropout(0.333)
		self.batch1 = nn.BatchNorm1d(n_hidden)  
		self.batch2 = nn.BatchNorm1d(n_hidden)  
		self.is_dropout = is_dropout
		self.is_batch = is_batch
		self.activation=activation

	def forward(self,x):
		x = self.layer1(x)
		if self.activation=='relu':
			x = torch.relu(x)
		elif self.activation=='gelu':
			x = torch.nn.functional.gelu(x)
		else:
			raise Exception('bad activation passed in')
		if self.is_dropout:
			x = self.drop1(x)
		if self.is_batch:
			x = self.batch1(x)

		x = self.layer2(x)
		if self.activation=='relu':
			x = torch.relu(x)
		elif self.activation=='gelu':
			x = torch.nn.functional.gelu(x)
		if self.is_dropout:
			x = self.drop2(x)
		if self.is_batch:
			x = self.batch2(x)

		x = self.layer3(x)
		return x

class Model_cnn(nn.Module):
	# quantile outputs are independent of eachother
	# I've largely copied the SurvMNISTNN model in
	# https://github.com/rajesh-lab/X-CAL/blob/master/models/simple_ff.py
	# but they add on a large number of FC layers at end (see appendix F2) which I don't think is helpful
	def __init__(self,input_dim,n_hidden,n_quantiles,is_dropout=False,is_batch=False):
		super().__init__()
		self.conv1 = nn.Conv2d(input_dim[0],64,kernel_size=(5,5),stride=(1, 1)) # n_channels, out_channels, kernel_size,
		self.conv2 = nn.Conv2d(64,128,kernel_size=(5,5))
		self.pool1 = nn.AvgPool2d(kernel_size=(2,2))
		self.conv3 = nn.Conv2d(128,256,kernel_size=(3,3))
		self.pool2 = nn.AvgPool2d(kernel_size=(2,2))
		self.layer3 = None # avoid specifying shape ahead of time
		self.drop1 = nn.Dropout2d(0.2)
		self.drop2 = nn.Dropout2d(0.2)
		self.batch1 = nn.BatchNorm2d(16)
		self.batch2 = nn.BatchNorm2d(32)
		self.is_dropout = is_dropout
		self.is_batch = is_batch
		self.n_quantiles = n_quantiles

	def forward(self,x):
		x = self.conv1(x)
		x = torch.relu(x)
		if self.is_dropout:
			x = self.drop1(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = torch.relu(x)
		if self.is_dropout:
			x = self.drop2(x)
		x = self.pool2(x)

		x = self.conv3(x)
		x = torch.relu(x)
	
		x = torch.flatten(x,start_dim=1, end_dim=- 1)
		if self.layer3 == None: # can avoid specifying shape ahead of time with this
			self.layer3 = nn.Linear(x.size(1), self.n_quantiles)
		x = self.layer3(x)
		return x


class Model_mlp_nocross(nn.Module):
	# NN with two relu hidden layers
	# quantile outputs now add consecutively on eachother, after passing through a softplus
	
	# it takes about twice as long as independent mlp to train
	# and doesn't perform performance any so..

	def __init__(self,input_dim,n_hidden,n_quantiles):
		super(Model_mlp_nocross,self).__init__()
		self.layer1 = nn.Linear(input_dim,n_hidden,bias=True)
		self.layer2 = nn.Linear(n_hidden,n_hidden,bias=True)

		# instead of layer3 as in mlp, we have a single layer stacking on top of eachother
		self.layers = nn.ModuleList() # need to do this instead of usual python list!
		self.n_quantiles = n_quantiles
		for i in range(n_quantiles):
			self.layers.append(nn.Linear(n_hidden,1,bias=True))

	def forward(self,x):
		x = self.layer1(x)
		x = torch.relu(x)
		x = self.layer2(x)
		x = torch.relu(x)
		# x = self.layer3(x)

		outs_i = self.layers[0](x)
		outs = outs_i
		outs_prev = outs_i
		for i in range(1,self.n_quantiles):
			outs_i = nn.functional.softplus(self.layers[i](x)) + outs_prev
			outs = torch.cat([outs, outs_i],dim=-1)
			outs_prev = outs_i
		return outs


def crossing_loss(y_pred):
	# crossing loss
	# y_pred is size (n_batch, n_quantiles)
	# where adjacent quantiles are consecutive
	# https://stats.stackexchange.com/questions/249874/the-issue-of-quantile-curves-crossing-each-other
	loss_cross = 0
	margin=0.1
	alpha=10
	diffs = y_pred[:,1:-1] - y_pred[:,:-2] # we would like diffs all to be +ve if not crossing
	loss_cross = alpha*torch.mean(torch.maximum(torch.tensor(0.0), margin -diffs))
	return loss_cross


def quantile_loss(y_pred, y_true, cen_indicator, taus, taus_torch, IS_FORCE_ALL_OBS):
	# standard checkmark / tilted pinball loss used for quantile regression
	# but we also pass in cen_indicator and avoid calculating this over those datapoints

	if IS_FORCE_ALL_OBS: # for naive baseline 'all observed'
		cen_indicator = cen_indicator*0. # force all datapoints to be observed

	tau_block = taus_torch.repeat((cen_indicator.shape[0],1)) # need this stacked in shape (n_batch, n_quantiles)
	loss = torch.sum((cen_indicator<1)*(y_pred  - y_true)*((1-tau_block)-1.*(y_pred<y_true)),dim=1)
	loss = torch.mean(loss)
	# I thought about whether this should be /N (mean as here), or /N_observed, torch.sum(loss)/torch.sum(cen_indicator<1)
	# and same for censored loss
	# I confirmed it definitely should all be /N, so fine to use mean
	return loss


def cqrnn_loss(y_pred, y_true, cen_indicator, taus, taus_torch, IS_USE_CENSOR_LOSS, IS_USE_CROSS_LOSS, IS_FORCE_ALL_OBS, y_max):
	# this is CQRNN loss as in paper
	# y_pred is shape (n_batch, n_quantiles)
	# y_true is shape (n_batch, 1)
	# cen_indicator is shape (n_batch, 1)

	# we've taken care to implement the loss without for loops, so things can be parallelised quickly
	# but the downside is that this becomes harder to read and match up with the description in the paper
	# so we also include cqrnn_loss_slowforloops() 
	# just note they both do the same thing

	# 1) first do all observed data points, censored loss not required
	# 2) second do all censored observations, no observed points

	# use detach to figure out where to block
	# first figure out closest quantile (do for all observations)
	y_pred_detach = y_pred.detach()
	# do we need detach()? yes I think so, otherwise loss is affected, though it's argmin so gradients prob don't flow anyway

	# should do this outside loss really and subselect here if needed
	tau_block = taus_torch.repeat((cen_indicator.shape[0],1)) # need this stacked in shape (n_batch, n_quantiles), 

	loss_obs = quantile_loss(y_pred, y_true, cen_indicator, taus, taus_torch, IS_FORCE_ALL_OBS)

	# add in crossing loss
	if IS_USE_CROSS_LOSS:
		loss_obs+=crossing_loss(y_pred)

	if IS_USE_CENSOR_LOSS==False or IS_FORCE_ALL_OBS==True:  # for naive baseline 'excl. observed' and 'all observed'
		return loss_obs # this only returns loss over observed data points

	# use argmin to get nearest quantile
	torch_abs = torch.abs(y_true - y_pred_detach[:,:-1]) # ignore the final quantile, which represents 1.0, so use [:-1]
	estimated_quantiles = torch.max(tau_block[:,:-1]*(torch_abs==torch.min(torch_abs,dim=1).values.view(torch_abs.shape[0],1)),dim=1).values
		
	# compute weights, eq 11, portnoy 2003
	# want weights to be in shape (batch_size x n_quantiles-1)
	weights = (tau_block[:,:-1]<estimated_quantiles.reshape(-1,1))*1. + (tau_block[:,:-1]>=estimated_quantiles.reshape(-1,1))*(tau_block[:,:-1]-estimated_quantiles.reshape(-1,1))/(1-estimated_quantiles.reshape(-1,1))

	# now compute censored loss using 
	# weight* censored value, + (1-weight)* fictionally large value
	y_max=y_max # just use a really high value, larger than any data point we'll see
	loss_cens = torch.sum((cen_indicator>0) * \
						  (weights * (y_pred[:,:-1]  - y_true)*((1-tau_block[:,:-1])-1.*(y_pred[:,:-1]<y_true)) + \
				          (1-weights)*(y_pred[:,:-1]  - y_max )*((1-tau_block[:,:-1])-1.*(y_pred[:,:-1]<y_max ))) \
				          ,dim=1)
	# could drop *(y_pred[:,:-1]<y_max) as this will always be true, but incl. for completeness
	loss_cens = torch.mean(loss_cens)

	return loss_obs + loss_cens


def neocleous_loss(y_pred, y_true, cen_indicator, taus, taus_torch, estimated_quantiles, IS_USE_CROSS_LOSS, y_max):
	# this is neocleous loss for sequential grid
	# also used in uncond_cqrnn baseline
	# y_pred is shape (n_batch, 1)
	# y_true is shape (n_batch, 1)
	# cen_indicator is shape (n_batch, 1)
	# estimated_quantiles is shape (n_batch, 1)

	# we now no longer need to bootstrap estimated_quantiles, these are passed in directly
	# taus and taus_torch is only one element
	# note that weight estimation doesn't actually matter when below the current quantile, so we init these to 1 in training loop
	# we'll ignore the crossing loss when model passed in only predicts one quantile, as for neocleous
	# as would require predictions from quantile below, and checking against them, which is a faff
	# don't compare methods with crossing loss turned on

	# should do this outside loss really and subselect here if needed
	tau_block = taus_torch.repeat((cen_indicator.shape[0],1)) # need this stacked in shape (n_batch, n_quantiles), 

	loss_obs = quantile_loss(y_pred, y_true, cen_indicator, taus, taus_torch, IS_FORCE_ALL_OBS=False)

	# add in crossing loss
	if IS_USE_CROSS_LOSS:
		loss_obs+=crossing_loss(y_pred)

	# compute weights, eq 11, portnoy 2003
	# want weights to be in shape (batch_size x n_quantiles-1)
	weights = (tau_block<estimated_quantiles.reshape(-1,1))*1. + (tau_block>=estimated_quantiles.reshape(-1,1))*(tau_block-estimated_quantiles.reshape(-1,1))/(1-estimated_quantiles.reshape(-1,1))

	# now compute censored loss using 
	# weight* censored value, + (1-weight)* fictionally large value
	loss_cens = torch.sum((cen_indicator>0) * \
						  (weights * (y_pred  - y_true)*((1-tau_block)-1.*(y_pred<y_true)) + \
				          (1-weights)*(y_pred  - y_max )*((1-tau_block)-1.*(y_pred<y_max ))) \
				          ,dim=1)
	# could drop *(y_pred[:,:-1]<y_max) as this will always be true, but incl. for completeness
	loss_cens = torch.mean(loss_cens)

	# print(loss_cens)
	return loss_obs + loss_cens


def cqrnn_loss_slowforloops(y_pred, y_true, cen_indicator, taus, IS_USE_CENSOR_LOSS, n_quantiles):
	# this is CQRNN loss as in paper
	# but this is the slower non parallelised version, with for loops 
	# (added for readability only, not recommended for use, esp. with large grid size!)

	# censored tilted multi loss
	loss_obs = 0
	
	y_pred_detach = y_pred.detach()
	estimated_quantiles = torch.zeros_like(cen_indicator)

	# tp combining first and second steps into same loop for efficiency
	for i in range(n_quantiles-1):
		tau = taus[i]
		loss_obs += torch.mean((cen_indicator<1)*(y_pred[:,i].reshape(-1,1)  - y_true)*((1-tau)-1.*(y_pred[:,i].reshape(-1,1)<y_true)))
		# (note this returns lower bound on quantile, not argmin)
		estimated_quantiles[y_true > y_pred_detach[:,i].reshape(-1,1)] = taus[i]

	if not IS_USE_CENSOR_LOSS:
		return loss_obs # this only computes loss over observed data points

	# compute weights, eq 11, portnoy 2003
	weights = torch.zeros(cen_indicator.size(0), n_quantiles-1) # shape batch_size, n_quantiles 
	for i in range(n_quantiles-1):
		tau = taus[i]
		weights[:,i]  = (tau < estimated_quantiles.reshape(-1))*1.+ (tau >= estimated_quantiles.reshape(-1)) * (tau - estimated_quantiles.reshape(-1)) / (1 - estimated_quantiles.reshape(-1))

	# now compute censored loss using 
	# weight* censored value, + (1-weight)* fictionally large value
	y_max=99.
	loss_cens = 0
	for i in range(n_quantiles-1):
		tau = taus[i]
		loss_cens += torch.mean((cen_indicator>0)*(weights[:,i].reshape(-1,1)*(y_pred[:,i].reshape(-1,1)  - y_true)*((1-tau)-1.*(y_pred[:,i].reshape(-1,1)<y_true)) \
			+ (1-weights[:,i].reshape(-1,1))*(y_pred[:,i].reshape(-1,1)  - y_max)*((1-tau)-1.*(y_pred[:,i].reshape(-1,1)<y_max))))

	return loss_obs + loss_cens



def deepquantreg_loss(y_pred, y_true, cen_indicator, global_kmf_torch_batch, taus, taus_torch, n_quantiles, IS_USE_CROSS_LOSS):
	# loss function used in, jia & jeong, DeepQuantreg 2021
	# only ever computing loss over observed data points 
	# this is weighted by global KM estimator (done outside of this loss function)

	# loss = 0
	# first compute weights via section 2.3
	weights = torch.zeros(cen_indicator.size(0), 1) # shape batch_size, 1 
	weights[:,:] = 1/(global_kmf_torch_batch+1e-2)

	# for vectorised form, we need weights to be in shape (batch_size x n_quantiles-1)
	# note the weights don't change by the quantile
	weights_block = weights.repeat((1, n_quantiles-1))
	tau_block = taus_torch.repeat((cen_indicator.shape[0],1))

	loss = torch.sum(weights_block*(cen_indicator<1)*(y_pred[:,:-1]  - y_true)*((1-tau_block[:,:-1])-1.*(y_pred[:,:-1]<y_true)),dim=1)
	loss = torch.mean(loss) 

	# add in crossing loss
	if IS_USE_CROSS_LOSS:
		loss+=crossing_loss(y_pred)

	return loss


def deepquantreg_loss_slowforloops(y_pred, y_true, cen_indicator, global_kmf_torch_batch, taus, n_quantiles):
	# loss function used in, jia & jeong, DeepQuantreg 2021
	# this is a slower non-optimised version using for loops

	loss = 0
	# first compute weights via section 2.3
	# note the weights don't change by the quantile
	weights = torch.zeros(cen_indicator.size(0), 1) # shape batch_size, 1 
	weights[:,:] = 1/(global_kmf_torch_batch+1e-2)
	# weights[:,:] = 1/(global_kmf_torch_batch)

	
	# do all observed data points, censored loss not required
	# this follows eq 7, though we don't use logs here
	for i in range(n_quantiles-1):
		tau = taus[i]
		# masked version of check fn, eg from eq 3, DeepQuantreg
		loss += torch.mean(weights[:].reshape(-1,1)*(cen_indicator<1)*(y_pred[:,i].reshape(-1,1)  - y_true)*((1-tau)-1.*(y_pred[:,i].reshape(-1,1)<y_true)))
	
	return loss


def lognorm_loss(y_pred, y_true, cen_indicator):
	# y_pred is now shape batch_size, 2
	# representing mean and stddev of log normal dist
	# (but stddev needs transforming w softplus)
	# logT = N(mean, stddev^2)
	# for observed data points, want to minimise -logpdf
	# for censored, want to minimise -logcdf
	# this helped: https://github.com/rajesh-lab/X-CAL/blob/master/models/loss/mle.py

	mean = y_pred[:,0]
	soft_fn = nn.Softplus()
	stddev = soft_fn(y_pred[:,1])

	pred_dist = torch.distributions.LogNormal(mean,stddev)

	logpdf = torch.diagonal(pred_dist.log_prob(y_true))
	cdf = torch.diagonal(pred_dist.cdf(y_true))
	logsurv = torch.log(1.0-cdf+1e-4)

	loglike = torch.mean((cen_indicator<1)*logpdf + (cen_indicator>0)*logsurv)
	loss = -loglike

	return loss

def lognorm_to_quantiles(model, x_in, taus, n_quantiles):
	# convert from model outputting mean and stddev of lognorm
	# to quantile estimates

	# predict parameters with torch
	y_pred = model(x_in)
	mean = y_pred[:,0].detach().cpu().numpy()
	soft_fn = nn.Softplus()
	stddev = soft_fn(y_pred[:,1]).detach().cpu().numpy()

	# note that scipy uses shape and scale param, while torch uses mean and stddev
	# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
	y_pred = []
	for i in range(n_quantiles-1):
		y_pred.append(lognorm.ppf(taus[i], stddev, scale=np.exp(mean)))
	y_pred.append(lognorm.ppf(0.9999, stddev, scale=np.exp(mean))) # add a dummy at end as we usually have this
	y_pred = np.array(y_pred).T

	return y_pred



