import matplotlib.pyplot as plt
import numpy as np
import torch
from lifelines import KaplanMeierFitter
import time
import pickle
import os

from utils import *
from models import *
from datasets import *
from hyperparams import *
from algorithms import *

# this file contains main script used for experiments

# make fonts compatible with submissions
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
plt.rcParams['text.usetex'] = True

start_time = time.time()

n_runs = 10
is_show_input_graph=False # whether to plot graph over input space
is_save_input_graph=False # whether to save said input graph
is_show_pred_graph=False # whether to plot graph of predictions
is_KM_graph=False # whether to plot unconditional KM graph
is_save_results=False # whether to save results as pickle file
IS_USE_CROSS_LOSS=False # if True add in part to loss that penalises if crossed quantiles
use_gpu=False # try to use GPU (if available)
is_verbose=1 # whether to print stuff out
my_rand_seed=111

# set up specific gpu if wanted
os.environ['CUDA_VISIBLE_DEVICES']='1'
if torch.cuda.is_available() and use_gpu:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
print('using device:', device)

dataset_str_list = []

# fully synthetic 1D
dataset_str_list+=['Gaussian_linear'] # Norm linear
dataset_str_list+=['Gaussian_nonlinear'] # Norm non-linear
dataset_str_list+=['Exponential'] # Exponential
dataset_str_list+=['Weibull'] # Weibull
dataset_str_list+=['Gaussian_uniform'] # Norm uniform
dataset_str_list+=['LogNorm'] # LogNorm

# multi-dim fully synthetic
dataset_str_list+=['Gaussian_Uniform4D_v1'] # Norm medium
dataset_str_list+=['GaussianUniform4D_v1_heavy'] # Norm heavy
dataset_str_list+=['GaussianUniform4D_v1_light'] # Norm light
dataset_str_list+=['Gaussian_Uniform4D_v2'] # Norm same
dataset_str_list+=['LogNorm_v1'] # LogNorm medium
dataset_str_list+=['LogNorm_v1_heavy'] # LogNorm heavy
dataset_str_list+=['LogNorm_v1_light'] # LogNorm light
dataset_str_list+=['LogNorm_v2'] # LogNorm same

# real data, synth censoring:
dataset_str_list+=['Housing']
dataset_str_list+=['Protein']
dataset_str_list+=['Wine']
dataset_str_list+=['PHM']
dataset_str_list+=['SurvMNISTv2']

# real data, real censoring
dataset_str_list+=['METABRICv2']
dataset_str_list+=['WHAS']
dataset_str_list+=['SUPPORT']
dataset_str_list+=['GBSG']
dataset_str_list+=['TMBImmuno']
dataset_str_list+=['BreastMSK']
dataset_str_list+=['LGGGBM']

for dataset_str in dataset_str_list:
	print('\n\n|||||||||||| dataset_str',dataset_str,'\n')

	results_all=OrderedDict()
	method_list=[]
	method_list += ['cqrnn']
	method_list += ['excl_censor']
	method_list += ['deepquantreg']
	method_list += ['lognorm']
	method_list += ['neocleous'] # this is sequential grid algorithm for NNs
	for loss_str in method_list:

		# load hyperparameters
		hyp = get_hyperparams(dataset_str, loss_str)
		model_str = hyp['model_str']
		x_range = hyp['x_range']
		n_data = hyp['n_data']
		n_test = hyp['n_test']
		n_show = hyp['n_show']
		test_propotion = hyp['test_propotion']
		n_quantiles = hyp['n_quantiles']
		n_hidden = hyp['n_hidden']
		n_epochs = hyp['n_epochs']
		batch_size = hyp['batch_size']
		learning_rate = hyp['learning_rate']
		weight_decay = hyp['weight_decay']
		y_max = hyp['y_max']
		is_dropout = hyp['is_dropout']
		is_batch = hyp['is_batch']
		activation = hyp['activation']
		
		if is_verbose==1: print(hyp)

		# default for CQRNN
		IS_USE_CENSOR_LOSS = True # if False, you only train on non-censored data
		IS_FORCE_ALL_OBS = False # if True then loss treats all data points as observed
		if loss_str == 'all_obs':
			IS_USE_CENSOR_LOSS = False 
			IS_FORCE_ALL_OBS = True
		elif loss_str == 'excl_censor':
			IS_USE_CENSOR_LOSS = False 
			IS_FORCE_ALL_OBS = False 

		taus = np.linspace(1/n_quantiles,1,n_quantiles) # doesn't use the last one
		taus_assess = [0.1,0.5,0.9] # these are only used to compute MSE
		if n_quantiles<10:
			taus[0] = 0.1
			taus[-2] = 0.9
		if n_quantiles==6 and is_show_input_graph: # just for 1D plots
			taus[1] = 0.3
			taus[3] = 0.7
		if is_verbose==1:
			print('loss_str: ', loss_str)
			print('taus: ', taus[:-1])
		taus = np.array([round(x,3) for x in taus])

		for run in range(n_runs):
			if is_verbose==1:
				print('\n---- method',loss_str,' run', run+1, 'of',n_runs)
			else:
				print('---- method',loss_str,' run', run+1, 'of',n_runs,end='\r')
			
			# set random seeds
			np.random.seed(run+my_rand_seed)
			rand_in = run+my_rand_seed
			torch.manual_seed(run+my_rand_seed)

			# put this inside loop to ensure get different data sample and/or test/train split, but same across methods
			mydataset = get_dataset(dataset_str)

			if mydataset.synth_target==True and mydataset.synth_cen==True:
				x_train, tte_train, cen_train, y_train, cen_indicator, obs_indicator = generate_data_synthtarget_synthcen(n_data, mydataset, x_range, is_censor=True)
				# draw 1 set of test data from observed distribution only
				x_test, tte_test, cen_test, y_test, cen_indicator_test, obs_indicator_test = generate_data_synthtarget_synthcen(n_test, mydataset, x_range, is_censor=False)
				# draw a second set of test data censored as for training distribution
				x_test_cens, tte_test_cens, cen_test_cens, y_test_cens, cen_indicator_test_cens, obs_indicator_test_cens = generate_data_synthtarget_synthcen(n_test, mydataset, x_range, is_censor=True)
			elif mydataset.synth_target==False and mydataset.synth_cen==True:
				data_train, data_test, data_test_cens = generate_data_realtarget_synthcen(mydataset, test_propotion, rand_in)
				x_train, tte_train, cen_train, y_train, cen_indicator, obs_indicator = data_train
				x_test, tte_test, cen_test, y_test, cen_indicator_test, obs_indicator_test = data_test
				x_test_cens, tte_test_cens, cen_test_cens, y_test_cens, cen_indicator_test_cens, obs_indicator_test_cens = data_test_cens
				n_data = x_train.shape[0]
				n_test = x_test.shape[0]
			elif mydataset.synth_target==False and mydataset.synth_cen==False:
				# for these datasets, data_test will be same as data_test_cens, but we leave it here so don't have to edit metrics etc
				data_train, data_test, data_test_cens = generate_data_realtarget_realcen(mydataset, test_propotion, rand_in)
				x_train, _, _, y_train, cen_indicator, obs_indicator = data_train
				x_test, _, _, y_test, cen_indicator_test, obs_indicator_test = data_test
				x_test_cens, _, _, y_test_cens, cen_indicator_test_cens, obs_indicator_test_cens = data_test_cens 
				n_data = x_train.shape[0]
				n_test = x_test.shape[0]

			if hyp['y_max'] == '1.2x':
				hyp['y_max'] = 1.2*y_train.max()
				y_max = hyp['y_max']

			# cen_indicator = 1 if censored else 0, while obs_indicator = 1 if observed else 0
			input_dim = mydataset.input_dim
			if type(input_dim)==int or len(input_dim)<=1: # flag for whether is image dataset
				is_img = False
			else:
				is_img = True

			if is_verbose==1:
				print('proportion censored:',round(cen_indicator.mean(),3))

			if loss_str not in ['neocleous','lognorm']:
				if model_str == 'cnn_small':
					model = Model_cnn(input_dim,n_hidden,n_quantiles,is_dropout,is_batch)
				elif model_str == 'linear':
					model = Model_linear(input_dim,n_hidden,n_quantiles)
				elif model_str == 'mlp':
					model = Model_mlp(input_dim,n_hidden,n_quantiles,is_dropout,is_batch,activation)
				elif model_str == 'mlp_nocross':
					model = Model_mlp_nocross(input_dim,n_hidden,n_quantiles)

				if is_img and model_str not in ['cnn_small']:
					raise Exception('error, pls use CNN with image data')

				model.to(device)

			x_train_torch = torch.tensor(x_train).float().to(device)
			x_test_torch = torch.tensor(x_test).float().to(device)
			x_test_cens_torch = torch.tensor(x_test_cens).float().to(device)
			if input_dim==1:
				x_grid = np.linspace(x_test_cens.min()-0.1,x_test_cens.max()+0.1,100).reshape([1,-1]).T
				x_grid_torch = torch.tensor(x_grid).float().to(device)
			y_train_torch = torch.tensor(y_train.reshape([1,-1]).T).float().to(device)
			cen_indicator_torch = torch.tensor(cen_indicator.reshape([1,-1]).T).float().to(device)
			taus_torch = torch.tensor(taus).reshape([1,-1]).float().to(device) # shape (1,n_quantiles)

			# need these K-M weights, eg for deepquantreg loss
			kmf = KaplanMeierFitter()
			kmf.fit(y_train, event_observed=obs_indicator) 
			global_kmf = np.array(kmf.predict(y_train)) # now get global KM estimator at each datapoint (obs and cens, but will only use obs later!)
			global_kmf_torch = torch.tensor(global_kmf.reshape([1,-1]).T).float().to(device)

			if is_KM_graph:
				fig, ax = plt.subplots(1,1)
				ax.scatter(y_train, global_kmf)
				ax.scatter(y_train, cen_indicator,color='r',s=5)
				ax.set_xlabel('y_train')
				ax.set_ylabel('global_kmf')
				fig.show()

			if is_verbose==1: print('max quantile defined on global km curve:',round(1-global_kmf.min(),3))

			if loss_str in ['cqrnn','all_obs','excl_censor','deepquantreg','ucond_cqrnn']:
				train_start_time = time.time()
				model = train_loop_single_models(model, x_train_torch, y_train_torch, cen_indicator_torch, global_kmf_torch, learning_rate, 
								weight_decay, n_epochs, batch_size, loss_str, taus, taus_torch, n_data, n_quantiles,
								IS_USE_CENSOR_LOSS, IS_USE_CROSS_LOSS, IS_FORCE_ALL_OBS, y_max, device, is_verbose)

				train_end_time = time.time()

				# forward passes for metrics and visualisation
				model.eval()
				test_start_time = time.time()
				if input_dim==1:
					y_grid_preds = model(x_grid_torch).detach().cpu().numpy()
				y_preds_train = model(x_train_torch).detach().cpu().numpy()  
				y_preds_test = model(x_test_torch).detach().cpu().numpy()
				y_preds_test_cens = model(x_test_cens_torch).detach().cpu().numpy()
				test_end_time = time.time()

			if loss_str == 'neocleous':
				# Neocleous et al. 2006, grid based method
				# first set up n_quantile models, each with one output
				models_neo=[]
				for i in range(n_quantiles):
					models_neo.append(Model_mlp(input_dim,n_hidden,1,is_dropout,is_batch))
					models_neo[-1].tau=taus[i]

				train_start_time = time.time()
				model = train_loop_neocleous(models_neo, x_train_torch, y_train_torch, cen_indicator_torch, global_kmf_torch, learning_rate, 
					weight_decay, n_epochs, batch_size, loss_str, taus, taus_torch, n_data, n_quantiles,
					IS_USE_CENSOR_LOSS, IS_USE_CROSS_LOSS, IS_FORCE_ALL_OBS, y_max, device, is_verbose)
				train_end_time = time.time()

				# forward passes for metrics and visualisation
				for i in range(n_quantiles):
					models_neo[i].eval()
				test_start_time = time.time()
				if input_dim==1:
					y_grid_preds = np.array([models_neo[i](x_grid_torch).detach().numpy() for i in range(n_quantiles)]).squeeze().T
				y_preds_train = np.array([models_neo[i](x_train_torch).detach().numpy() for i in range(n_quantiles)]).squeeze().T
				y_preds_test = np.array([models_neo[i](x_test_torch).detach().numpy() for i in range(n_quantiles)]).squeeze().T
				y_preds_test_cens = np.array([models_neo[i](x_test_cens_torch).detach().numpy() for i in range(n_quantiles)]).squeeze().T
				test_end_time = time.time()

			if loss_str == 'lognorm':
				# this is trained via maximum likelihood
				if model_str == 'cnn_small':
					model = Model_cnn(input_dim,n_hidden,2,is_dropout,is_batch)
				elif model_str == 'linear':
					model = Model_linear(input_dim,n_hidden,2)
				elif model_str == 'mlp':
					model = Model_mlp(input_dim,n_hidden,2,is_dropout,is_batch)
				model.to(device)

				train_start_time = time.time()
				model = train_loop_single_models(model, x_train_torch, y_train_torch, cen_indicator_torch, global_kmf_torch, learning_rate, 
								weight_decay, n_epochs, batch_size, loss_str, taus, taus_torch, n_data, n_quantiles,
								IS_USE_CENSOR_LOSS, IS_USE_CROSS_LOSS, IS_FORCE_ALL_OBS, y_max, device, is_verbose)
				train_end_time = time.time()

				# forward passes for metrics and visualisation
				model.eval()
				test_start_time = time.time()
				if input_dim==1:
					y_grid_preds = lognorm_to_quantiles(model, x_grid_torch, taus, n_quantiles)
				y_preds_train = lognorm_to_quantiles(model, x_train_torch, taus, n_quantiles)
				y_preds_test = lognorm_to_quantiles(model, x_test_torch, taus, n_quantiles)
				y_preds_test_cens = lognorm_to_quantiles(model, x_test_cens_torch, taus, n_quantiles)
				test_end_time = time.time()

			# could fix crossed quantiles here, e.g. 
			# y_preds_train = fn_uncross_quantiles(y_preds_train)
			# y_preds_test = fn_uncross_quantiles(y_preds_test)
			# y_preds_test_cens = fn_uncross_quantiles(y_preds_test_cens)

			if input_dim==1 and (is_show_input_graph==True or is_save_input_graph==True):
				plot_save_name = '1Dinput_' + dataset_str + '_' + loss_str +'_v01'
				visualise_1d(x_train, y_train, cen_indicator, x_grid, y_grid_preds, mydataset, taus, n_data, n_show, n_quantiles, is_save_input_graph, plot_save_name, title_in=loss_str+', '+dataset_str)
			elif is_show_input_graph and not is_img:
				visualise_nd(x_train, y_train, cen_indicator, y_preds_train, taus, n_data, n_show, n_quantiles, input_dim, title_in=loss_str+', '+dataset_str)

			# compute and print metrics
			if is_verbose==1:
				print('loss_str', loss_str, ', dataset_str', dataset_str, ', n_data',n_data,', n_quantiles',n_quantiles, ', dataset_str', dataset_str)
			# train
			metrics_train = compute_metrics(x_train, y_preds_train, y_train, cen_indicator, obs_indicator, mydataset, taus, taus_assess, n_quantiles, 'train', dataset_str, hyp, is_print=False)
			# test observed dist only
			metrics_test = compute_metrics(x_test, y_preds_test, y_test, cen_indicator_test, obs_indicator_test, mydataset, taus, taus_assess, n_quantiles, 'test', dataset_str, hyp, is_print=True)
			# test mixed observed & censored
			metrics_test_cens = compute_metrics(x_test_cens, y_preds_test_cens, y_test_cens, cen_indicator_test_cens, obs_indicator_test_cens, mydataset, taus, taus_assess, n_quantiles, 'test_cens', dataset_str, hyp, is_print=False)

			# quick plot of test truth vs predctions
			if is_show_pred_graph:
				colors = plt.cm.cool(np.linspace(0,1,n_quantiles-1))
				fig, ax = plt.subplots(1,1)
				ax.plot(np.array([0,1]),np.array([0,1]),color='k',alpha=0.5,lw=0.5)
				for i in range(n_quantiles-1):
					# ax.scatter(y_preds_test[:500,int(n_quantiles/2)], y_test[:500]+np.random.normal(0,0.01,size=500),color='r',s=10,alpha=0.1)
					ax.scatter(y_preds_test[:,i], y_test+np.random.normal(0,0.01,size=y_test.shape[0]),color=colors[i],s=5,alpha=0.01)
				ax.set_xlabel('y_preds_test')
				ax.set_ylabel('y_test')
				fig.show()

			metrics_time = OrderedDict()
			metrics_time['training_time'] = train_end_time - train_start_time
			metrics_time['test_fwdpass_time'] = test_end_time - test_start_time
			metrics_time['highest_def_quantile'] = 1-global_kmf.min()
			metrics_time['y_train_max'] = y_train.max()
			metrics_time['y_train_90'] = np.quantile(y_train,0.9)

			results_all = append_metrics(metrics_train, 'train', loss_str, results_all)
			results_all = append_metrics(metrics_test, 'test', loss_str, results_all)
			results_all = append_metrics(metrics_test_cens, 'test_cens', loss_str, results_all)
			results_all = append_metrics(metrics_time, 'time', loss_str, results_all)

		# optionally print off parameter count
		if is_verbose and False:
			if loss_str == 'neocleous':
				param_count = sum(p.numel() for p in model[0].parameters() if p.requires_grad)*len(model)
			else:
				param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
			print('\nparam_count:', param_count, dataset_str, loss_str)


	print('\n\nresults over', n_runs, 'runs, for dataset:', dataset_str)
	for method in method_list:
		print('\n===',method)
		for name_in in ['test','test_cens','time']:
			print('\n-',name_in)
			if name_in=='test':
				# metric_list = ['mmse_mean','Dcal_nocens','q_metric_selected','quantile_selected_low','quantile_selected_mid','quantile_selected_hig']
				metric_list = ['TQMSE','UQL','UnDCal','CensDCal']
				if mydataset.synth_cen==False:
					continue # don't assess in this case
			elif name_in=='test_cens':
				# metric_list = ['cindex_median','Dcal_cens']
				metric_list = ['cindex_median','CensDCal']
			elif name_in=='time':
				metric_list = ['training_time','test_fwdpass_time']
			for key in metric_list:
				key_res = method + '_' + name_in + '_' + key
				try:
					print(key, 'avg\t', round(np.mean(results_all[key_res]),3), '\t(s.e.', round(np.std(results_all[key_res])/np.sqrt(len(results_all[key_res])),3),')')
				except:
					pass

	# so results_all contains results over multiple runs, for various methods, for one dataset, for one hyperparam setting
	if is_save_results:
		file_name = 'results_run_xyz_'+dataset_str+'_ep'+str(n_epochs)+'_run'+str(n_runs)+'.p'
		save_path = os.path.join(path_results,file_name)
		pickle.dump([dataset_str_list, method_list, results_all], open(save_path,'wb'))
		print('\nsaved results_all at',save_path)
		# [dataset_str_list, method_list, results_all] = pickle.load(open(save_path,'rb'))

end_time = time.time()

print('\ntime (secs) taken:',round(end_time-start_time,3),'\n')
