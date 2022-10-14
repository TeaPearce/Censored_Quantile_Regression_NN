import numpy as np
from lifelines.utils import concordance_index
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
import os

# this file contains metrics and some plotting functions plus general housekeeping

# make fonts compatible with submissions
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
plt.rcParams['text.usetex'] = True

# useful paths -- I'm expecting scripts to be run from within 02_code
path_cd = os.getcwd()
path_project = os.path.dirname(path_cd) # up one level
path_data = os.path.join(path_project,'02_datasets')
path_results = os.path.join(path_project,'03_results')
path_plots = os.path.join(path_project,'04_plots')
path_csvs = os.path.join(path_project,'05_csv')

# what type each dataset is
synth_tar_list = ['Gaussian_linear', 'Gaussian_nonlinear','Exponential','Weibull','LogNorm','Gaussian_uniform','GaussianUniform4D_v1_heavy','Gaussian_Uniform4D_v1','GaussianUniform4D_v1_light','Gaussian_Uniform4D_v2','LogNorm_v1_heavy','LogNorm_v1','LogNorm_v1_light','LogNorm_v2']
synth_cen_list = ['Housing', 'Protein','Wine','PHM','SurvMNISTv2']
real_cen_list = ['METABRICv2', 'WHAS','SUPPORT','GBSG','TMBImmuno','BreastMSK','LGGGBM']
dataset_dict = {'synth_1D':synth_tar_list[:6],'synth_ND':synth_tar_list[6:],'synth_cens':synth_cen_list,'real_cens':real_cen_list}

def quantiles_to_median(y_preds, taus):
	# return something prediction when tau=0.5
	# if doesn't exist raise an error
	eps = 1e-3
	if ((taus > 0.5-eps)*(taus < 0.5+eps)).sum() != 1:
		raise Exception('no median available')
	median_idx = np.argmax(np.array((taus > 0.5-eps)*(taus < 0.5+eps)))
	return  y_preds[:,median_idx]

def quantiles_to_mean(y_preds, taus):
	# compute an approximation of the mean
	# thought about using this for the c-index calculation, but didn't use in the end
	# favouring the simpler median which performed v similarly

	# ignore tails of the distribution that are outside of those we're predicting
	# we will place a point mass in the centre of each bin between quantiles
	# we ignore the first and last bins as they are undefined
	taus_mass = taus[1:-1] - taus[:-2]
	taus_mass = taus_mass*1/taus_mass.sum() # inflate for tails
	y_preds_mid = (y_preds[:,:-2] + y_preds[:,1:-1])/2
	y_mean = np.sum(y_preds_mid*taus_mass,axis=-1)
	return y_mean

# metrics
def MMSE_fn(y_true, y_pred_in, cen_indicator=None):
	# mean squared error
	# y_pred_in is 1D array
	return np.mean(np.square(y_true[cen_indicator.flatten()==0] - y_pred_in[cen_indicator.flatten()==0]))

def MMAE_fn(y_true, y_pred_in, cen_indicator=None):
	# mean absolute error
	# y_pred_in is 1D array
	return np.mean(np.abs(y_true[cen_indicator.flatten()==0] - y_pred_in[cen_indicator.flatten()==0]))

def quantile_loss_fn(y_true, y_pred_in, taus, q=0.5, cen_indicator=None):
	# does quantile pinball/checkmark loss
	# this is UQL in paper

	# find which quantile we predicted is closest
	idx = np.argmin(np.abs(taus-q))
	eps = 1e-3
	if np.abs(taus[idx] - q) > eps:
		print('\nquantile_loss_fn may have an error, asked for quantile ',q,' only have',taus[idx], 'available')

	y_pred_in = y_pred_in[:,idx].flatten()
	# y_pred_in is 1D array
	# loss = torch.sum((cen_indicator<1)*(y_pred  - y_true)*((1-tau_block)-1.*(y_pred<y_true)),dim=1)
	obs_idx = cen_indicator.flatten()==0
	# multiply by 2 to get
	return np.mean((y_true[obs_idx] - y_pred_in[obs_idx])*(q-1.*(y_pred_in[obs_idx]>y_true[obs_idx])))


def quantile_compare_fn(x, y_preds_in, mydataset, taus, q=0.9):
	# compare true quantile vs predicted quantile using MSE
	# this is TQMSE in paper
	# y_preds_in is 2D array (n_samples, n_taus)

	# find index of quantile which is closest to target
	idx = np.argmin(np.abs(taus-q))
	eps = 1e-3
	if np.abs(taus[idx] - q) > eps:
		print('\nmaybe an error, asked for quantile ',q,' only have',taus[idx], 'available')

	# get ground truth
	y_true = mydataset.get_quantile_truth(x.squeeze(),q)

	# mean squared error
	return np.mean(np.square(y_preds_in[:,idx] - y_true))


def compute_metrics(x_in, y_preds, y_true, cen_indicator, obs_indicator, mydataset, taus, taus_assess, n_quantiles, name_in, dataset_str, hyp, is_print=True):
	# a handy collector function that computes all metrics at once
	# not all of the metrics computed will be useful for every input, but we save them all anyway
	# for example it will never make sense to look at ECE for the training data since censoring is always present

	mean_pred = quantiles_to_mean(y_preds, taus)
	median_pred = quantiles_to_median(y_preds, taus)

	metrics = OrderedDict()
	# metrics['name'] = name_in
	metrics['hyp'] = hyp
	metrics['xshape'] = x_in.shape
	metrics['dataset_str'] = dataset_str
	# metrics['n_quantiles'] = n_quantiles
	metrics['prop_censored'] = np.mean(cen_indicator)
	metrics['mmse_mean'] = MMSE_fn(y_true, mean_pred, cen_indicator)
	# metrics['mmse_median'] = MMSE_fn(y_true, median_pred, cen_indicator) # mmse but with median
	metrics['mmae_median'] = MMAE_fn(y_true, median_pred, cen_indicator)
	metrics['quantile_selected_low'] = np.mean([quantile_loss_fn(y_true, y_preds, taus, q, cen_indicator) for q in [0.1]])
	metrics['quantile_selected_mid'] = np.mean([quantile_loss_fn(y_true, y_preds, taus, q, cen_indicator) for q in [0.5]])
	metrics['quantile_selected_hig'] = np.mean([quantile_loss_fn(y_true, y_preds, taus, q, cen_indicator) for q in [0.9]])
	metrics['UQL'] = metrics['quantile_selected_low'] + metrics['quantile_selected_mid'] + metrics['quantile_selected_hig'] 
	# concordance_index() lifelines docs: 1 if observed, 0 if not. 
	metrics['cindex_mean'] = concordance_index(y_true, mean_pred, obs_indicator)
	metrics['cindex_median'] = concordance_index(y_true, median_pred, obs_indicator)
	if mydataset.synth_target==True:
		metrics['q_metric_avg'] = np.mean([quantile_compare_fn(x_in, y_preds, mydataset, taus, q) for q in taus[:-1]])
		metrics['q_metric_selected'] = np.mean([quantile_compare_fn(x_in, y_preds, mydataset, taus, q) for q in taus_assess])
		metrics['TQMSE'] = metrics['q_metric_selected']

	# calibration plots
	# for each quantile predicting, find proportion of observed datapoints smaller than what we predicted
	# measure absolute difference between identity line
	# this turns out to not be quite correct unless data is sampled from true observed distribution with no censoring
	calibration_data = []
	calibration_data.append([0.0, 0.0])
	for i in range(n_quantiles-1):
		cal_prop_target = taus[i]
		cal_prop_larger = np.mean(y_preds[cen_indicator.flatten()==0,i] > y_true[cen_indicator.flatten()==0])
		calibration_data.append([cal_prop_target, cal_prop_larger])
	calibration_data.append([1.0, 1.0])
	calibration_data = np.array(calibration_data)
	metrics['greater_ECE'] = np.mean(np.abs(calibration_data[:,0]-calibration_data[:,1]))

	# now look at how much falls in between each quantile
	# D-calibration, taken from X-CAL: Explicit calibration for survival analysis
	dcal_data=[]
	for i in range(calibration_data.shape[0]-1):
		target = calibration_data[i+1,0] - calibration_data[i,0]
		captured = calibration_data[i+1,1] - calibration_data[i,1]
		dcal_data.append([target,captured])
	dcal_data = np.array(dcal_data)
	
	metrics['ECE'] = np.sum(np.abs(dcal_data[:,0]-dcal_data[:,1]))
	metrics['Dcal_nocens'] = 100*np.sum(np.square(dcal_data[:,0]-dcal_data[:,1])) # this is UnDCal from paper
	metrics['Dcal_nocens_data'] = dcal_data
	metrics['UnDCal'] = metrics['Dcal_nocens']

	# now D-calibration for censored data
	# X-CAL: Explicit calibration for survival analysis, eq 9 and 10
	# for each data point, find predicted quantile
	diffs = y_preds[:,:-1] - np.expand_dims(y_true,axis=1)
	closest_q_idx = np.argmin(np.abs(diffs),axis=1)
	closest_q = []
	for i in range(y_true.shape[0]):
		closest_q.append(taus[closest_q_idx[i]])
	closest_q=np.array(closest_q)

	# for each bin
	dcal_data_cens=[]
	for i in range(n_quantiles):
		if i>0:
			a=taus[i-1]
		else:
			a=0.
		b = taus[i]
		# find how many obs and cens data points in between a and b
		# going off eq 9 and 10 here
		if b<1.:
			smaller_b = y_preds[cen_indicator.flatten()==0,i] > y_true[cen_indicator.flatten()==0]
			smaller_b_cens = y_preds[cen_indicator.flatten()==1,i] > y_true[cen_indicator.flatten()==1]
		else:
			smaller_b = 1e9 > y_true[cen_indicator.flatten()==0]
			smaller_b_cens = 1e9 > y_true[cen_indicator.flatten()==1]
		if a>0.:
			larger_a = y_preds[cen_indicator.flatten()==0,i-1] <= y_true[cen_indicator.flatten()==0]
			larger_a_cens = y_preds[cen_indicator.flatten()==1,i-1] <= y_true[cen_indicator.flatten()==1]
			smaller_a_cens = y_preds[cen_indicator.flatten()==1,i-1] > y_true[cen_indicator.flatten()==1]
		else:
			larger_a = -1e9 <= y_true[cen_indicator.flatten()==0]
			larger_a_cens = -1e9 <= y_true[cen_indicator.flatten()==1]
			smaller_a_cens = -1e9 > y_true[cen_indicator.flatten()==1]

		fallwithin = smaller_b*larger_a
		fallwithin_cens = smaller_b_cens*larger_a_cens
		cens_part1 = fallwithin_cens*(b-closest_q[cen_indicator.flatten()==1])/(1-closest_q[cen_indicator.flatten()==1])
		cens_part2 = smaller_a_cens*(b-a)/(1-closest_q[cen_indicator.flatten()==1])

		total_points = fallwithin.sum() + cens_part1.sum() + cens_part2.sum()
		prop_captured = total_points/y_true.shape[0]
		# print(total_points, b-a,prop_captured)
		# total_points_sum+=total_points
		# print(a, 'larger_a',1-larger_a.sum()/y_true.shape[0])

		dcal_data_cens.append([b-a,prop_captured])
		
	dcal_data_cens = np.array(dcal_data_cens)

	metrics['Dcal_cens'] = 100*np.sum(np.square(dcal_data_cens[:,0]-dcal_data_cens[:,1]))  # this is CensDCal from paper
	# metrics['Dcal_cens'] = 100*np.sum(np.square(dcal_data_cens[:-1,0]-dcal_data_cens[:-1,1]))
	metrics['Dcal_cens_data'] = dcal_data_cens
	# warning! there is a very slight difference between cens and nocens Dcal because of crossing quantiles!
	# Dcal_nocens is slightly preferred
	metrics['CensDCal'] = metrics['Dcal_cens']

	if False:
		# could display calibration curve
		fig, ax = plt.subplots(1,1,figsize=(5,5))
		ax.grid()
		ax.plot([0,1],[0,1],color='k',alpha=1.,ls='--')
		ax.plot(calibration_data[:,0],calibration_data[:,1],color='fuchsia')
		ax.scatter(calibration_data[:,0],calibration_data[:,1],color='fuchsia')
		ax.plot(taus,dcal_data_cens[:,0],color='red',label='target prop.')
		ax.scatter(taus,dcal_data_cens[:,0],color='red')
		ax.plot(taus,dcal_data[:,1],color='skyblue',label='non cens prop.')
		ax.scatter(taus,dcal_data[:,1],color='skyblue')
		ax.plot(taus,dcal_data_cens[:,1],color='gray',label='censor prop.')
		ax.scatter(taus,dcal_data_cens[:,1],color='gray')
		ax.set_xlim([0,1])
		ax.set_ylim([0,1])
		ax.set_xlabel(r'Target proportion, $\tau$')
		ax.set_ylabel('Empirical proportion')
		ax.set_title(name_in)
		fig.legend()
		fig.show()

	metrics['calibration_data'] = calibration_data
	return metrics


def append_metrics(metrics, name_in, loss_str, results_dict):
	# appends metrics to an OrderedDict object
	for key in metrics:
		key_res = loss_str + '_' + name_in + '_' + key
		if key_res not in results_dict.keys():
			results_dict[key_res] = [metrics[key]]
		else:
			results_dict[key_res].append(metrics[key])

	return results_dict


def visualise_1d(x_train, y_train, cen_indicator, x_grid, y_grid_preds, mydataset, taus, n_data, n_show, n_quantiles, is_save_input_graph, save_name=None, title_in=None):
	# y_grid_preds is numpy array of predictions over grid
	# if n_show<n_data: print('\nonly displaying subset of data points!')
	# lw_main = 2.
	lw_main = 6.
	fig, ax = plt.subplots(1,1,figsize=(8,5))
	ax.scatter(x_train[:n_show][cen_indicator[0,:n_show] == 0], y_train[:n_show][cen_indicator[0,:n_show] == 0],color='g',marker='+',s=70,label='Observed')
	ax.scatter(x_train[:n_show][cen_indicator[0,:n_show] == 1], y_train[:n_show][cen_indicator[0,:n_show] == 1],color='g',marker='^',s=50,label='Censored')
	if mydataset.synth_target==True:
		# ax.plot(x_grid, mydataset.get_quantile_truth(x_grid,q=taus[0]),label='True q\'s',color='k',linestyle=':', alpha=1.,lw=lw_main)
		ax.plot(x_grid, mydataset.get_quantile_truth(x_grid,q=taus[0]),label='True quantiles',color='k',linestyle=':', alpha=1.,lw=lw_main)
		# ax.plot(x_grid, mydataset.get_mean_truth(x_grid),label='true mean',color='orange',linestyle='--', alpha=1.,lw=4.)
		# ax.plot(x_grid, mydataset.get_censored_quantile_truth(x_grid,q=taus[0]),label='Censor q\'s',color='gray',linestyle=':', alpha=1.,lw=lw_main/2)
	# ax.plot(x_grid, quantiles_to_median(y_preds, taus),color='r',label='est. median',linestyle='--', alpha=1.,lw=4.)
	# ax.plot(x_grid, quantiles_to_mean(y_grid_preds, taus),color='r',label='est. mean',linestyle='--', alpha=1.,lw=4.)
	colors = plt.cm.cool(np.linspace(0,1,n_quantiles-1))
	for i in range(n_quantiles-1):
		ax.plot(x_grid, y_grid_preds[:,i],label='Est. quantile '+str(round(taus[i],3)),color=colors[i], alpha=1.,lw=lw_main)
	if mydataset.synth_target==True:
		for i in range(n_quantiles-1):
			ax.plot(x_grid, mydataset.get_quantile_truth(x_grid,q=taus[i]),color='k',linestyle=':', alpha=1.,lw=lw_main)
			# ax.plot(x_grid, mydataset.get_censored_quantile_truth(x_grid,q=taus[i]),color='gray',linestyle=':', alpha=1.,lw=lw_main/2)
	if title_in is not None and is_save_input_graph==False:
		ax.set_title(title_in)
	ax.set_ylim([y_train.min()*0.909,y_train.max()*1.1])
	if is_save_input_graph:
		# matplotlib.rcParams.update({'font.size': 16})
		# ax.set_xlim([0,2])
		ax.set_xlim([-0.1,2.1])
		# ax.xaxis.set_major_locator(plt.MaxNLocator(4))
		ax.set_xticks([0,1,2])
		ax.set_yticks([])
		ax.set_xticks([])
		fig.savefig(os.path.join(path_plots,'test_'+save_name+'.pdf'),dpi=100,format='pdf', bbox_inches='tight', pad_inches=0.1)
		print('saved input graph to:', os.path.join(path_plots,save_name+'.pdf'))

		if False:
			# save separate legend as individual image
			# https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
			# then create a new image
			# adjust the figure size as necessary
			fig_leg = plt.figure(figsize=(2, 3))
			ax_leg = fig_leg.add_subplot(111)
			ax_leg.legend(*ax.get_legend_handles_labels(), loc='center',frameon=False)
			ax_leg.axis('off')
			fig_leg.savefig(os.path.join(path_plots,'1Dinput_legend_01.pdf'),dpi=100,format='pdf', bbox_inches='tight', pad_inches=0.)

	fig.legend()
	fig.show()
	return

def visualise_nd(x_train, y_train, cen_indicator, y_preds_train, taus, n_data, n_show, n_quantiles, input_dim, title_in=None):
	# as a sense check for higher dim inputs, we plot x along each input axis (effectively marginalising over other dims)
	# quantile predictions are also overlaid
	# if n_show<n_data: print('\nonly displaying subset of data points!')
	fig, ax = plt.subplots(min(input_dim,10),1,figsize=(8,10))
	for j in range(min(input_dim,10)):
		ax[j].scatter(x_train[:n_show,j][cen_indicator[0,:n_show] == 0], y_train[:n_show][cen_indicator[0,:n_show] == 0],color='g',marker='+',s=30,label='observed',alpha=0.2)
		ax[j].scatter(x_train[:n_show,j][cen_indicator[0,:n_show] == 1], y_train[:n_show][cen_indicator[0,:n_show] == 1],color='g',marker='^',s=20,label='censored',alpha=0.2)
		colors = plt.cm.cool(np.linspace(0,1,n_quantiles-1))
		for i in range(n_quantiles-1):
			ax[j].scatter(x_train[:n_show,j], y_preds_train[:n_show,i],label='est. q '+str(round(taus[i],3)),color=colors[i],s=2)
		ax[j].set_ylim([y_train.min()*0.909,y_train.max()*1.1])
	if title_in is not None:
		ax[0].set_title(title_in)
	ax[0].legend()
	fig.show()
	return
