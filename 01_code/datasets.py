import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import os
import pandas as pd
import torchvision 
import torch
import h5py
from collections import defaultdict

from utils import *

# this file contains synthetic dataset classes
# we use 'target' to refer to true target
# 'cen' or 'censor' to refer to censoring time
# and then 'y' is minimum of those, which we actually get to train on

def get_dataset(dataset_str):
	if dataset_str == 'Gaussian_linear':
		mydataset = GaussianLinear()
	elif dataset_str == 'Gaussian_nonlinear':
		mydataset = GaussianNonLinear()
	elif dataset_str == 'Gaussian_nonlinear_censnon':
		mydataset = GaussianNonLinearCensNon()
	elif dataset_str == 'Gaussian_const':
		mydataset = GaussianConst()
	elif dataset_str == 'Gaussian_same':
		mydataset = GaussianSame()
	elif dataset_str == 'Weibull':
		mydataset = Weibull()
	elif dataset_str == 'Exponential':
		mydataset = Exponential()
	elif dataset_str == 'LogNorm':
		mydataset = LogNorm()
	elif dataset_str == 'Gaussian_uniform':
		mydataset = GaussianUniform_v1()
	elif dataset_str == 'Gaussian_Uniform4D_v1':
		mydataset = GaussianUniform4D_v1()
	elif dataset_str == 'GaussianUniform4D_v1_heavy':
		mydataset = GaussianUniform4D_v1_heavy()
	elif dataset_str == 'GaussianUniform4D_v1_light':
		mydataset = GaussianUniform4D_v1_light()
	elif dataset_str == 'Gaussian_Uniform4D_v2':
		mydataset = GaussianUniform4D_v2()
	elif dataset_str == 'LogNorm_v1':
		mydataset = LogNorm_v1()
	elif dataset_str == 'LogNorm_v1_heavy':
		mydataset = LogNorm_v1_heavy()
	elif dataset_str == 'LogNorm_v1_light':
		mydataset = LogNorm_v1_light()
	elif dataset_str == 'LogNorm_v2':
		mydataset = LogNorm_v2()
	elif dataset_str == 'Housing':
		mydataset = Housing()
	elif dataset_str == 'Protein':
		mydataset = Protein()
	elif dataset_str == 'Wine':
		mydataset = Wine()
	elif dataset_str == 'PHM':
		mydataset = PHM()
	elif dataset_str == 'SurvMNISTv2':
		mydataset = SurvMNISTv2()
	elif dataset_str == 'METABRICv2':
		mydataset = METABRICv2()
	elif dataset_str == 'TMBImmuno':
		mydataset = TMBImmuno()
	elif dataset_str == 'BreastMSK':
		mydataset = BreastMSK()
	elif dataset_str == 'LGGGBM':
		mydataset = LGGGBM()
	elif dataset_str == 'WHAS':
		mydataset = WHAS()
	elif dataset_str == 'GBSG':
		mydataset = GBSG()
	elif dataset_str == 'SUPPORT':
		mydataset = SUPPORT()
	else:
		raise Exception(dataset_str+'dataset not defined')
	return mydataset

def generate_data_synthtarget_synthcen(n_data, mydataset, x_range=[0,2], is_censor=True):
	# is_censor=False allows to only draw data from observed dist
	# this fn is used if we're generating everything, data and censoring

	# sample x
	x = np.random.uniform(x_range[0],x_range[1],size=(n_data,mydataset.input_dim))

	# compute target
	target = mydataset.get_observe_times(x).flatten()

	# compute censor
	if is_censor:
		cen = mydataset.get_censor_times(x).flatten()
	else:
		cen = target+100. # otherwise make censored times larger than observed

	# y = min(target,cen)
	y, cen_indicator, obs_indicator = mydataset.process_censor_observe(target, cen)

	return x, target, cen, y, cen_indicator, obs_indicator

def generate_data_realtarget_synthcen(mydataset, test_propotion, rand_in):
	x_train, x_test, y_target_train, y_target_test, cen_train, cen_test_cens = mydataset.get_data(test_propotion,rand_in)

	# censoring for training dataset
	y_train, cen_indicator, obs_indicator = mydataset.process_censor_observe(y_target_train, cen_train)
	data_train = (x_train, y_target_train, cen_train, y_train, cen_indicator, obs_indicator)

	# test set without censoring
	cen_test = y_target_test+100. # manufacture this so no datapoints censored
	y_test, cen_indicator_test, obs_indicator_test = mydataset.process_censor_observe(y_target_test, cen_test)
	data_test = (x_test, y_target_test, cen_test, y_test, cen_indicator_test, obs_indicator_test)

	# test with censoring
	y_test_cens, cen_indicator_test_cens, obs_indicator_test_cens = mydataset.process_censor_observe(y_target_test, cen_test_cens)
	data_test_cens = (x_test, y_target_test, cen_test_cens, y_test_cens, cen_indicator_test_cens, obs_indicator_test_cens)
		
	return data_train, data_test, data_test_cens

def generate_data_realtarget_realcen(mydataset, test_propotion, rand_in):
	x_train, x_test, y_train, y_test, cen_indicator_train, cen_indicator_test = mydataset.get_data(test_propotion,rand_in)

	data_train = (x_train, y_train, y_train, y_train, cen_indicator_train, np.abs(cen_indicator_train-1))
	data_test = (x_test, y_test, y_test, y_test, cen_indicator_test, np.abs(cen_indicator_test-1))
	data_test_cens = data_test # these are equivalent for this task
	
	return data_train, data_test, data_test_cens

class DataSet:
	def process_censor_observe(self, target, cen):
		y = np.minimum(cen,target)
		cen_indicator = np.array([cen<target])*1 # 1 if censored else 0
		obs_indicator = np.array([cen>=target])*1 # 1 if observed else 0
		return y, cen_indicator, obs_indicator

class RealTargetSyntheticCensor(DataSet):
	def __init__(self):
		self.synth_target=False
		self.synth_cen=True
		pass
	def vis_data(self):
		nshow=1000
		fig, ax = plt.subplots(self.input_dim,1)
		for i in range(self.input_dim):
			ax[i].scatter(self.df.data[:nshow,i], self.df.target[:nshow],s=6, alpha=0.5)
		fig.show()
	def get_data(self, test_propotion,rand_in):

		# get a random test/train split
		self.x_train, self.x_test, self.y_target_train, self.y_target_test = train_test_split(self.df.data, self.df.target, test_size=test_propotion, random_state=rand_in)

		# !!! subselect for development speed
		# n_subselect=50000000
		# self.x_train, self.x_test, self.y_target_train, self.y_target_test = self.x_train[:n_subselect], self.x_test[:n_subselect], self.y_target_train[:n_subselect], self.y_target_test[:n_subselect]

		# generate artificial censoring times
		self.cen_train_cens = self.generate_censoring(self.x_train)
		self.cen_test_cens = self.generate_censoring(self.x_test)

		self.y_target_train, self.y_target_test, self.cen_train_cens, self.cen_test_cens = self.y_target_train.flatten(), self.y_target_test.flatten(), self.cen_train_cens.flatten(), self.cen_test_cens.flatten()

		return self.x_train, self.x_test, self.y_target_train, self.y_target_test, self.cen_train_cens, self.cen_test_cens

	def generate_censoring(self, x):
		if len(x.shape)==2:
			return np.random.uniform(x[:,0]*0.+self.df.target.min(), x[:,0]*0.+np.quantile(self.df.target,0.9)) # could also do this
		else:
			return np.random.uniform(x[:,0,0,0]*0.+self.df.target.min(), x[:,0,0,0]*0.+self.df.target.max()*1.5) # independent uniform 30% censoring


class Housing(RealTargetSyntheticCensor):
	def __init__(self):
		super().__init__() 
		self.input_dim=8

		# https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
		self.df = fetch_california_housing()

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.df.data)
		self.df.data = x_scaler.transform(self.df.data)
		y_scaler = preprocessing.StandardScaler().fit(self.df.target.reshape(-1, 1))
		self.df.target = y_scaler.transform(self.df.target.reshape(-1, 1))

		# clip outliers
		x_lim = 5
		self.df.data = np.clip(self.df.data,-x_lim,x_lim)
		self.df.target = np.clip(self.df.target,-x_lim,x_lim)
		self.df.target-=self.df.target.min()-1e-1 # adjust so no negative times

		if False:
			# optionally visualise
			self.vis_data()
		return

class Protein(RealTargetSyntheticCensor):
	def __init__(self):
		super().__init__() 
		# https://openml.org/search?type=data&status=active&qualities.NumberOfInstances=between_10000_100000&id=42903
		# https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
		self.input_dim=9
		self.df = fetch_openml(name='physicochemical-protein')

		self.df.target = self.df.data.pop('RMSD')

		self.df.target = np.array(self.df.target)
		self.df.data = np.array(self.df.data)

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.df.data)
		self.df.data = x_scaler.transform(self.df.data)
		y_scaler = preprocessing.StandardScaler().fit(self.df.target.reshape(-1, 1))
		self.df.target = y_scaler.transform(self.df.target.reshape(-1, 1))

		# clip outliers
		x_lim = 5
		self.df.data = np.clip(self.df.data,-x_lim,x_lim)
		self.df.target = np.clip(self.df.target,-x_lim,x_lim)
		self.df.target-=self.df.target.min()-1e-1 # adjust so no negative times

		if False: # optionally visualise
			self.vis_data()
		return


class Wine(RealTargetSyntheticCensor):
	def __init__(self):
		super().__init__() 
		# https://www.openml.org/search?type=data&status=active&qualities.NumberOfClasses=lte_1&id=287
		self.input_dim=11
		self.df = fetch_openml(name='wine_quality')

		# self.df.target = self.df.data.pop('RMSD')

		self.df.target = np.array(self.df.target)
		self.df.data = np.array(self.df.data)

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.df.data)
		self.df.data = x_scaler.transform(self.df.data)
		y_scaler = preprocessing.StandardScaler().fit(self.df.target.reshape(-1, 1))
		self.df.target = y_scaler.transform(self.df.target.reshape(-1, 1))

		# clip outliers
		x_lim = 5
		self.df.data = np.clip(self.df.data,-x_lim,x_lim)
		self.df.target = np.clip(self.df.target,-x_lim,x_lim)
		self.df.target-=self.df.target.min()-1e-1 # adjust so no negative times

		if False: # optionally visualise
			self.vis_data()
		return
		
class PHM(RealTargetSyntheticCensor):
	def __init__(self):
		super().__init__() 
		self.input_dim=21
		self.df = fetch_openml(name='NASA_PHM2008')

		self.df.target = np.array(self.df.target)
		self.df.data = np.array(self.df.data)

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.df.data)
		self.df.data = x_scaler.transform(self.df.data)
		y_scaler = preprocessing.StandardScaler().fit(self.df.target.reshape(-1, 1))
		self.df.target = y_scaler.transform(self.df.target.reshape(-1, 1))

		# clip outliers
		x_lim = 5
		self.df.data = np.clip(self.df.data,-x_lim,x_lim)
		self.df.target = np.clip(self.df.target,-x_lim,x_lim)
		self.df.target-=self.df.target.min()-1e-1 # adjust so no negative times

		if False: # optionally visualise
			self.vis_data()
		return

class SurvMNISTv2(RealTargetSyntheticCensor):
	def __init__(self):
		super().__init__() 
		self.input_dim=(1,28,28)
		# self.df.data, self.df.target = 
		
		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'02_datasets')

		# download this. must use version torchvision==0.9.1 to get processed folder
		# https://github.com/pytorch/vision/issues/4685
		transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
													torchvision.transforms.Normalize((0.5,), (0.5,)),
													])
		trainset = torchvision.datasets.MNIST(path_data,download=True, train=True, transform=transform)
		# trainset is of shape (60000 , 2), with x and y (int)
		# each x example is (1, 28, 28)

		# we'll match structure of other RealTargetSyntheticCensor datasets by basing around a df object
		class Object(object):
			pass
		self.df = Object()

		# load whole dataset
		self.df.data, self.df.class_ = torch.load(os.path.join(os.path.join(os.path.join(path_data,'MNIST'),'processed'),'training.pt'))
		self.df.data, self.df.class_ = self.df.data.numpy(), self.df.class_.numpy() # this is also dumb but needed for censoring gen
		self.df.data = np.expand_dims(self.df.data,1) # add an extra channel

		# we now generate targets and censoring as per 
		# App D.2 https://arxiv.org/pdf/2101.05346.pdf
		# basically, each class is assigned a risk group, which has an associated risk score
		# this parameterises a gamma dist from which targets are drawn
		risk_list = [11.25, 2.25, 5.25, 5.0, 4.75, 8.0, 2.0, 11.0, 1.75, 10.75]
		var_list = [0.1, 0.5, 0.1, 0.2, 0.2, 0.2, 0.3, 0.1, 0.4, 0.6]
		# var_list = [1e-3]*10
		risks_mean = np.zeros_like(self.df.class_)+0.9
		risks_var = np.zeros_like(self.df.class_)+0.9
		for i in range(10):
			risks_mean[self.df.class_==i] = risk_list[i]
			risks_var[self.df.class_==i] = var_list[i]

		self.df.target = np.random.gamma(shape=np.square(risks_mean)/risks_var,scale=1/(risks_mean/risks_var)) 
		# self.df.target = np.random.gamma(shape=np.square(risks_mean)/var,scale=1/(risks_mean/var)) 
		# 1/ as diff param -- see https://en.wikipedia.org/wiki/Gamma_distribution

		# normalisation
		self.df.data = self.df.data/255
		self.df.target = self.df.target/10

		# could subselect for development
		# self.df.data, self.df.target = self.df.data[:10000], self.df.target[:10000]
		# self.df.data, self.df.target = self.df.data[:2000], self.df.target[:2000]

		if False: # optionally visualise
			self.vis_data()
		return

	def generate_censoring(self, x):
		return np.random.uniform(x[:,0,0,0]*0.+self.df.target.min(), x[:,0,0,0]*0.+np.quantile(self.df.target,0.9))
		# return np.random.uniform(x[:,0,0,0]*0.+self.df.target.min(), x[:,0,0,0]*0.+self.df.target.max()*1.1)
		# we draw censoring times uniformly between the minimum failure time in that split and the 90th percentile time, 
		# which, due to the particular failure distributions, resulted in about 50% censoring

class RealTargetRealCensor(DataSet):
	def __init__(self):
		self.synth_target=False
		self.synth_cen=False

	def vis_data(self):
		nshow=1000
		fig, ax = plt.subplots(self.input_dim,1)
		for i in range(self.input_dim):
			# ax[i].scatter(self.df.data[:nshow,i], self.df.target[:nshow,0],s=6, alpha=0.5)
			ax[i].scatter(self.data[:nshow,i][self.target[:nshow,1] == 0], self.target[:nshow,0][self.target[:nshow,1] == 0],color='g',marker='+',s=20,label='observed')
			ax[i].scatter(self.data[:nshow,i][self.target[:nshow,1] == 1], self.target[:nshow,0][self.target[:nshow,1] == 1],color='g',marker='^',s=10,label='censored')
		fig.show()

	def get_data(self, test_propotion, rand_in):

		# get a random test/train split
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=test_propotion, random_state=rand_in)

		# !!! subselect for development speed
		n_subselect=500000
		self.x_train, self.x_test, self.y_train, self.y_test = self.x_train[:n_subselect], self.x_test[:n_subselect], self.y_train[:n_subselect], self.y_test[:n_subselect]

		# note that y_train contains both target and censored columns, so we split here
		self.cen_train_indicator = self.y_train[:,1].reshape(1,-1)
		self.cen_test_indicator = self.y_test[:,1].reshape(1,-1)
		self.y_train = self.y_train[:,0].flatten()
		self.y_test = self.y_test[:,0].flatten()

		return self.x_train, self.x_test, self.y_train, self.y_test, self.cen_train_indicator, self.cen_test_indicator

class METABRICv2(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'02_datasets')
		# self.df = pd.read_csv(os.path.join(path_data,'metabric_IHC4_clinical_train_test.h5'))
		dataset_file = os.path.join(path_data,'metabric_IHC4_clinical_train_test.h5')

		# datasets and opening code borrowed from https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/utils.py
		datasets = defaultdict(dict)
		with h5py.File(dataset_file, 'r') as fp:
			for ds in fp:
				for array in fp[ds]:
					datasets[ds][array] = fp[ds][array][:]

		# for my exps, I merge test and train splits together and resplit later
		self.target = np.concatenate([datasets['test']['t'],datasets['train']['t']])
		self.data = np.concatenate([datasets['test']['x'],datasets['train']['x']],axis=0)
		self.event = np.concatenate([datasets['test']['e'],datasets['train']['e']]) # in this dataset 1=observed, 0=censored

		# we concat target with event
		self.target = np.stack([self.target,self.event]).T 

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)
		self.input_dim=self.data.shape[1]

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		if False: # optionally visualise
			self.vis_data()
		return

class WHAS(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'02_datasets')
		# self.df = pd.read_csv(os.path.join(path_data,'metabric_IHC4_clinical_train_test.h5'))
		dataset_file = os.path.join(path_data,'whas_train_test.h5')

		# datasets and opening code borrowed from https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/utils.py
		datasets = defaultdict(dict)
		with h5py.File(dataset_file, 'r') as fp:
			for ds in fp:
				for array in fp[ds]:
					datasets[ds][array] = fp[ds][array][:]

		# for my exps, I merge test and train splits together and resplit later
		self.target = np.concatenate([datasets['test']['t'],datasets['train']['t']])
		self.data = np.concatenate([datasets['test']['x'],datasets['train']['x']],axis=0)
		self.event = np.concatenate([datasets['test']['e'],datasets['train']['e']]) # in this dataset 1=observed, 0=censored

		# we concat target with event
		self.target = np.stack([self.target,self.event]).T 

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)
		self.input_dim=self.data.shape[1]

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()

		self.target[:,0]-=self.target[:,0].min()-1e-1 # adjust so no negative times

		if False: # optionally visualise
			self.vis_data()
		return


class GBSG(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 
		# note that the way I set up this exp is closer to Deep Extended Hazard Models for Survival Analysis
		# and not comparable w deepsurv paper since they use test/train split as per diff studies

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'02_datasets')
		# self.df = pd.read_csv(os.path.join(path_data,'metabric_IHC4_clinical_train_test.h5'))
		dataset_file = os.path.join(path_data,'gbsg_cancer_train_test.h5')

		# datasets and opening code borrowed from https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/utils.py
		datasets = defaultdict(dict)
		with h5py.File(dataset_file, 'r') as fp:
			for ds in fp:
				for array in fp[ds]:
					datasets[ds][array] = fp[ds][array][:]

		# for my exps, I merge test and train splits together and resplit later
		self.target = np.concatenate([datasets['test']['t'],datasets['train']['t']])
		self.data = np.concatenate([datasets['test']['x'],datasets['train']['x']],axis=0)
		self.event = np.concatenate([datasets['test']['e'],datasets['train']['e']]) # in this dataset 1=observed, 0=censored

		# we concat target with event
		self.target = np.stack([self.target,self.event]).T 

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)
		self.input_dim=self.data.shape[1]

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		# clip outliers
		x_lim = 5
		self.data = np.clip(self.data,-x_lim,x_lim)
		self.target = np.clip(self.target,-x_lim,x_lim)

		if False: # optionally visualise
			self.vis_data()
		return

class SUPPORT(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 
		# note that the way I set up this exp is closer to Deep Extended Hazard Models for Survival Analysis
		# and not comparable w deepsurv paper since they use test/train split as per diff studies

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'02_datasets')
		dataset_file = os.path.join(path_data,'support_train_test.h5')

		# datasets and opening code borrowed from https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/utils.py
		datasets = defaultdict(dict)
		with h5py.File(dataset_file, 'r') as fp:
			for ds in fp:
				for array in fp[ds]:
					datasets[ds][array] = fp[ds][array][:]

		# for my exps, I merge test and train splits together and resplit later
		self.target = np.concatenate([datasets['test']['t'],datasets['train']['t']])
		self.data = np.concatenate([datasets['test']['x'],datasets['train']['x']],axis=0)
		self.event = np.concatenate([datasets['test']['e'],datasets['train']['e']]) # in this dataset 1=observed, 0=censored

		# we concat target with event
		self.target = np.stack([self.target,self.event]).T 

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)
		self.input_dim=self.data.shape[1]

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		if False: # optionally visualise
			self.vis_data()
		return


class TMBImmuno(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 

		# http://www.cbioportal.org/study/clinicalData?id=tmb_mskcc_2018

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'02_datasets')
		self.df=pd.read_table(os.path.join(path_data,'tmb_immuno_mskcc.tsv'),sep='\t')

		event_arr = np.array(self.df['Overall Survival Status'])
		self.df['event'] = np.array([int(event_arr[i][0]) for i in range(event_arr.shape[0])])
		self.df['time'] = np.array(self.df['Overall Survival (Months)']).astype(np.float)

		self.df['age_new'] = np.array(self.df['Age at Which Sequencing was Reported (Days)'])
		sex_arr = np.array(self.df['Sex'])
		self.df['sex_new'] = np.array([1 if sex_arr[i] == 'Female' else 0 for i in range(sex_arr.shape[0])])

		# remove nans
		self.df.dropna(subset = ['event', 'time', 'age_new','sex_new','TMB (nonsynonymous)'],how='any',inplace=True)
	
		# use self.target instead of self.df.target to avoid pandas warning
		self.target = pd.concat([self.df.pop(x) for x in ['time','event']], axis=1)
		self.data = pd.concat([self.df.pop(x) for x in ['age_new','sex_new','TMB (nonsynonymous)']], axis=1)


		self.target = np.array(self.target)
		self.data = np.array(self.data)

		self.input_dim=self.data.shape[1]

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		# clip outliers
		x_lim = 5
		self.data = np.clip(self.data,-x_lim,x_lim)
		self.target = np.clip(self.target,-x_lim,x_lim)

		if False: # optionally visualise
			self.vis_data()
		return


class BreastMSK(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 

		# https://www.cbioportal.org/study/summary?id=breast_msk_2018

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'02_datasets')
		self.df=pd.read_table(os.path.join(path_data,'breast_msk_2018_clinical_data.tsv'),sep='\t')

		event_arr = np.array(self.df['Overall Survival Status'])
		self.df['event'] = np.array([int(event_arr[i][0]) for i in range(event_arr.shape[0])])
		self.df['time'] = self.df['Overall Survival (Months)']

		tmp_arr = np.array(self.df['ER Status of the Primary'])
		self.df['ER_new'] = np.array([1 if tmp_arr[i] == 'Positive' else 0 for i in range(tmp_arr.shape[0])])
		tmp_arr = np.array(self.df['Overall Patient HER2 Status'])
		self.df['HER2_new'] = np.array([1 if tmp_arr[i] == 'Positive' else 0 for i in range(tmp_arr.shape[0])])
		tmp_arr = np.array(self.df['Overall Patient HR Status'])
		self.df['HR_new'] = np.array([1 if tmp_arr[i] == 'Positive' else 0 for i in range(tmp_arr.shape[0])])

		# remove nans
		self.df.dropna(subset = ['event', 'time', 'ER_new','HER2_new','HR_new','Mutation Count','TMB (nonsynonymous)'],how='any',inplace=True)
	
		# use self.target instead of self.df.target to avoid pandas warning
		self.target = pd.concat([self.df.pop(x) for x in ['time','event']], axis=1)
		self.data = pd.concat([self.df.pop(x) for x in ['ER_new','HER2_new','HR_new','Mutation Count','TMB (nonsynonymous)']], axis=1)

		self.target = np.array(self.target)
		self.data = np.array(self.data)

		self.input_dim=self.data.shape[1]

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		# clip outliers
		x_lim = 5
		self.data = np.clip(self.data,-x_lim,x_lim)
		self.target = np.clip(self.target,-x_lim,x_lim)

		if False: # optionally visualise
			self.vis_data()
		return

class LGGGBM(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 

		# https://www.cbioportal.org/study/summary?id=lgggbm_tcga_pub

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'02_datasets')
		self.df=pd.read_table(os.path.join(path_data,'lgggbm_tcga_pub_clinical_data.tsv'),sep='\t')

		# remove nans
		self.df.dropna(subset = ['Overall Survival Status', 'Overall Survival (Months)', 'Diagnosis Age','Sex','Absolute Purity','Mutation Count','TMB (nonsynonymous)'],how='any',inplace=True)

		event_arr = np.array(self.df['Overall Survival Status'])
		self.df['event'] = np.array([int(event_arr[i][0]) for i in range(event_arr.shape[0])])
		self.df['time'] = np.array(self.df['Overall Survival (Months)']).astype(np.float)

		tmp_arr = np.array(self.df['Sex'])
		self.df['sex_new'] = np.array([1 if tmp_arr[i] == 'Female' else 0 for i in range(tmp_arr.shape[0])])
	
		# use self.target instead of self.df.target to avoid pandas warning
		self.target = pd.concat([self.df.pop(x) for x in ['time','event']], axis=1)
		self.data = pd.concat([self.df.pop(x) for x in ['Diagnosis Age','sex_new','Absolute Purity','Mutation Count','TMB (nonsynonymous)']], axis=1)

		self.target = np.array(self.target)
		self.data = np.array(self.data)

		self.input_dim=self.data.shape[1]

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		# clip outliers
		x_lim = 3
		self.data = np.clip(self.data,-x_lim,x_lim)
		# self.target = np.clip(self.target,-x_lim,x_lim)

		if False: # optionally visualise
			self.vis_data()
		return

class SyntheticDataSet(DataSet):
	def __init__(self):
		self.synth_target=True # whether timetoevent is synthetically generated
		self.synth_cen=True # whether censoring is synthetically generated
	def get_observe_times(self, x):
		pass
	def get_censor_times(self, x):
		pass
	def get_quantile_truth(self, x, q):
		pass
	def get_mean_truth(self, x):
		pass

class SyntheticDataSet1D(SyntheticDataSet):
	# parent class for 1D datasets
	def __init__(self):
		super().__init__() 
		self.input_dim=1

class GaussianUniform4D_v1(SyntheticDataSet):
	def __init__(self):
		super().__init__() 
		self.input_dim=4
		self.offset = 0
		x = np.random.uniform(0,2,size=(10000,self.input_dim))
		self.offset = -min(self.get_observe_times(x))+1	
		print('self.offset',self.offset)
		# self.y_max_cens = np.quantile(self.get_observe_times(x),0.95)
	def get_observe_times(self, x):
		# return np.clip(np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x)),1e-1,1000)
		return np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x))
	def get_censor_times(self, x):
		# return np.clip(np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x)),1e-1,1000)
		return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).mean()
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.uniform(loc=self.param1_cen(x),scale=self.param2_cen(x)).ppf(q)
	def param1_target(self,x):
		return x[:,0]*3+x[:,1]**2-x[:,2]**2+np.sin(x[:,3]*x[:,2]) + self.offset
	def param2_target(self,x):
		return x[:,0]*0 + 1.
	def param1_cen(self,x):
		return x[:,0]*0 
	def param2_cen(self,x):
		# return x[:,0]*0 + 12
		return x[:,0]*0 + 20

class GaussianUniform4D_v1_heavy(GaussianUniform4D_v1):
	# we lower the censoring point so that most of the target dist is undefined
	# this is heavy censoring
	def param2_cen(self,x):
		return x[:,0]*0 + 12

class GaussianUniform4D_v1_light(GaussianUniform4D_v1):
	# light censoring
	def param2_cen(self,x):
		return x[:,0]*0 + 40

class GaussianUniform4D_v2(GaussianUniform4D_v1):
	# same as for target dist
	def get_censor_times(self, x):
		return self.get_observe_times(x)

class LogNorm_v1_original(SyntheticDataSet):
	def __init__(self):
		super().__init__() 
		self.input_dim=8
		self.betas = np.random.uniform(-1,1,(self.input_dim,1)) # randomly draw these to figure out where to censor
		# self.betas = np.array([[0.8, 0.6, 0.4, 0.5, -0.3, 0.2, 0.0, -0.7]]).T
		x = np.random.uniform(0,2,size=(1000,self.input_dim))
		self.y_max_cens = np.quantile(self.get_observe_times(x),0.95)
		# print('y_max_cens',self.y_max_cens)
		# print(self.betas)
	def get_observe_times(self, x):
		return np.random.lognormal(mean=self.param1_target(x), sigma=self.param2_target(x))/10
		# the mean and standard deviation are not the values for the distribution itself, but of the underlying normal distribution it is derived from.
	def get_censor_times(self, x):
		return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		# https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).ppf(q)/10
	def get_mean_truth(self, x):
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).mean()/10
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.uniform(loc=self.param1_cen(x),scale=self.param2_cen(x)).ppf(q)
	def param1_target(self,x):
		return np.matmul(x, self.betas).squeeze()
	def param2_target(self,x):
		return x[:,0]*0 + 1.
	def param1_cen(self,x):
		return x[:,0]*0 
	def param2_cen(self,x):
		return x[:,0]*0 + self.y_max_cens

class LogNorm_v1(SyntheticDataSet):
	def __init__(self):
		super().__init__() 
		self.input_dim=8
		self.betas = np.array([[0.8, 0.6, 0.4, 0.5, -0.3, 0.2, 0.0, -0.7]]).T
		x = np.random.uniform(0,2,size=(1000,self.input_dim))
		self.y_max_cens = np.quantile(self.get_observe_times(x),0.95)
	def get_observe_times(self, x):
		return np.random.lognormal(mean=self.param1_target(x), sigma=self.param2_target(x))/10
		# the mean and standard deviation are not the values for the distribution itself, but of the underlying normal distribution it is derived from.
	def get_censor_times(self, x):
		return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		# https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).ppf(q)/10
	def get_mean_truth(self, x):
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).mean()/10
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.uniform(loc=self.param1_cen(x),scale=self.param2_cen(x)).ppf(q)
	def param1_target(self,x):
		return np.matmul(x, self.betas).squeeze()
	def param2_target(self,x):
		return x[:,0]*0 + 1.
	def param1_cen(self,x):
		return x[:,0]*0 
	def param2_cen(self,x):
		return x[:,0]*0 + 1.

class LogNorm_v1_light(LogNorm_v1):
	def param2_cen(self,x):
		return x[:,0]*0 + 3.5

class LogNorm_v1_heavy(LogNorm_v1):
	def param2_cen(self,x):
		return x[:,0]*0 + 0.4

class LogNorm_v2(LogNorm_v1):
	def get_censor_times(self, x):
		return self.get_observe_times(x)

class Gaussian(SyntheticDataSet1D):
	def get_observe_times(self, x):
		# return np.maximum(np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x)),0.1)
		return np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x))
	def get_censor_times(self, x):
		# return np.maximum(np.random.normal(loc=self.param1_cen(x), scale = self.param2_cen(x)),0.1)
		return np.random.normal(loc=self.param1_cen(x), scale = self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).mean()
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.norm(self.param1_cen(x),self.param2_cen(x)).ppf(q)

class GaussianLinear(Gaussian):
	def param1_target(self,x):
		return x*2+10
	def param2_target(self,x):
		return (x+1)/1
	def param1_cen(self,x):
		# return x*5+10
		return x*4+10
	def param2_cen(self,x):
		return (x*4+2)/5

class GaussianNonLinear(GaussianLinear):
	def param1_target(self,x):
		return x*np.sin(2*x)+10
	def param2_target(self,x):
		return (x+1)/2
	def param1_cen(self,x):
		return x*2+10
	def param2_cen(self,x):
		return 2

class GaussianNonLinear_v1(GaussianLinear):
	def param1_target(self,x):
		return x*np.sin(2*x)+10
	def param1_cen(self,x):
		return x*3+9
	def param2_cen(self,x):
		# return (x*4+2)/2
		return 2

class GaussianNonLinearCensNon(GaussianLinear):
	# as well as mean being non linear, also make variances non linear
	def param1_target(self,x):
		return 2*np.sin(2*x)+6
	def param2_target(self,x):
		return np.cos(3*x)+2
	def param1_cen(self,x):
		return 10-x*2
	def param2_cen(self,x):
		# return (x*4+2)/2
		return x*0.5+1

class GaussianConst(GaussianLinear):
	def param1_target(self,x):
		return x*0+10
	def param2_target(self,x):
		return x*0+2
	def param1_cen(self,x):
		return x*0+11
	def param2_cen(self,x):
		return x*0+2

class GaussianSame(GaussianLinear):
	def param1_target(self,x):
		return 2*x*np.sin(2*x)+10
	def param2_target(self,x):
		return (x+1)/2
	def param1_cen(self,x):
		return 2*x*np.sin(2*x)+10
	def param2_cen(self,x):
		return (x+1)/2

class GaussianUniform(SyntheticDataSet1D):
	# target is Gaussian, censored is Uniform
	def get_observe_times(self, x):
		# return np.maximum(np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x)),0.1)
		# return np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x))
		return np.clip(np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x)),0.1,10000)
	def get_censor_times(self, x):
		# return np.maximum(np.random.normal(loc=self.param1_cen(x), scale = self.param2_cen(x)),0.1)
		return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).mean()
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.uniform(loc=self.param1_cen(x),scale=self.param2_cen(x)).ppf(q)

class GaussianUniform_v1(GaussianUniform):
	def param1_target(self,x):
		return 2*x*np.cos(2*x)+13
	def param2_target(self,x):
		return (x**2+1/2)
	def param1_cen(self,x):
		return x*0
	def param2_cen(self,x):
		return x*0+18 # this is width

class Weibull(SyntheticDataSet1D):
	def __init__(self):
		super().__init__() 
		self.weibull_shape = 5 # =1 is exponential, could get fancy and make this depend on x
	def param1_target(self,x):
		return x*np.sin(2*(x-1))*4+10
	def param1_cen(self,x):
		# return (x+1)**2+10
		# return (x)*3+10
		return 20-x*3
	def get_observe_times(self, x):
		return self.param1_target(x)*np.random.weibull(a=self.weibull_shape,size=x.shape)
	def get_censor_times(self, x):
		return self.param1_cen(x)*np.random.weibull(a=self.weibull_shape,size=x.shape)
	def get_quantile_truth(self, x, q):
		return scipy.stats.weibull_min(c=self.weibull_shape,scale=self.param1_target(x)).ppf(q)
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.weibull_min(c=self.weibull_shape,scale=self.param1_cen(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.weibull_min(c=self.weibull_shape,scale=self.param1_target(x)).mean()

class Weibull_v1(SyntheticDataSet1D):
	def __init__(self):
		self.weibull_shape = 5 # =1 is exponential, could get fancy and make this depend on x
	def param1_target(self,x):
		return x*np.sin(2*x)+10
	def param1_cen(self,x):
		return x*3+10
	def get_observe_times(self, x):
		return self.param1_target(x)*np.random.weibull(a=self.weibull_shape,size=x.shape)
	def get_censor_times(self, x):
		return self.param1_cen(x)*np.random.weibull(a=self.weibull_shape,size=x.shape)
	def get_quantile_truth(self, x, q):
		return scipy.stats.weibull_min(c=self.weibull_shape,scale=self.param1_target(x)).ppf(q)
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.weibull_min(c=self.weibull_shape,scale=self.param1_cen(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.weibull_min(c=self.weibull_shape,scale=self.param1_target(x)).mean()

class Exponential(SyntheticDataSet1D):
	def param1_target(self,x):
		return x*2+4
	def param1_cen(self,x):
		# return x*3+10
		# return 20-x*3
		return 15-x*3
	def get_observe_times(self, x):
		return np.random.exponential(scale=self.param1_target(x))
	def get_censor_times(self, x):
		return np.random.exponential(scale=self.param1_cen(x))
	def get_quantile_truth(self, x, q):
		return scipy.stats.expon(scale=self.param1_target(x)).ppf(q)
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.expon(scale=self.param1_cen(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.expon(scale=self.param1_target(x)).mean()

class LogNorm(SyntheticDataSet1D):
	def get_observe_times(self, x):
		# return np.random.exponential(scale=self.param1_target(x))
		return np.random.lognormal(mean=self.param1_target(x), sigma=self.param2_target(x))
	def get_censor_times(self, x):
		# return np.random.exponential(scale=self.param1_cen(x))
		return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
		# return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		# return scipy.stats.expon(scale=self.param1_target(x)).ppf(q)
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).ppf(q)
		# return scipy.stats.lognorm(s=self.param2_target(x),scale=self.param1_target(x)).ppf(q)
	def get_censored_quantile_truth(self, x, q):
		# return scipy.stats.expon(scale=self.param1_cen(x)).ppf(q)
		return scipy.stats.uniform(loc=self.param1_cen(x),scale=self.param2_cen(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).mean()
	def param1_target(self,x):
		return (x-1)**2
	def param2_target(self,x):
		return x*0 + 1.
	def param1_cen(self,x):
		return x*0 
	def param2_cen(self,x):
		return x*0 + 10

