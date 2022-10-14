import numpy as np

# this file contains hyperparams for benchmarking experiments, after tuning

def get_hyperparams(dataset_str, loss_str):
	# create a dict of hyperparams
	hyp = {}

	# first specify defaults
	hyp = 	{'batch_size': 128,
			 'n_epochs': 100,
			 'learning_rate': 0.01,
			 'weight_decay': 1e-4,
			 'model_str': 'mlp', # what architecture to use
			 'is_dropout': False, # whether to use dropout
			 'is_batch': False, # whether to use batchnorm
			 'n_hidden': 100, # hidden nodes in NN
			 'n_quantiles': 10, # how many quantiles to predict
			 'test_propotion': 0.2, # proportion of dataset to use for test (real data)
			 'y_max': 99., # value to use for large pseudo datapoint in cqrnn loss
			 'n_data': 500, # train size (synth data)
			 'n_test': 1000, # test size (synth data)
			 'x_range': [0,2], # x range to sample data from (synth data)
			 'activation': 'relu' # activation, relu or gelu
			 }

	if dataset_str in ['Gaussian_linear','Gaussian_nonlinear', 'Exponential','Weibull','Gaussian_uniform','LogNorm']:
		hyp['n_data'] = 500
		hyp['n_test'] = 1000
		hyp['n_quantiles'] = 10 # set this to 10 for benchmarking experiment
		# hyp['n_quantiles'] = 6 # set this to 6 for 1D plots
		hyp['n_epochs'] = 100
		hyp['is_dropout'] = False
		hyp['activation'] = 'gelu' # makes visualisations slightly nicer
		if loss_str == 'cqrnn' or loss_str == 'neocleous':
			hyp['y_max'] = '1.2x'

	if dataset_str in ['Gaussian_Uniform4D_v1','GaussianUniform4D_v1_heavy', 'GaussianUniform4D_v1_light','Gaussian_Uniform4D_v2']:
		hyp['n_data'] = 2000
		hyp['n_test'] = 1000
		hyp['n_quantiles'] = 20
		hyp['n_epochs'] = 20
		hyp['is_dropout'] = False
		if loss_str == 'cqrnn' or loss_str == 'neocleous':
			hyp['y_max'] = '1.2x'

	if dataset_str in ['LogNorm_v1','LogNorm_v1_heavy','LogNorm_v1_light','LogNorm_v2']:
		hyp['n_data'] = 4000
		hyp['n_test'] = 1000
		hyp['n_quantiles'] = 20
		hyp['n_epochs'] = 10
		hyp['is_dropout'] = False
		if loss_str == 'cqrnn' or loss_str == 'neocleous':
			hyp['y_max'] = '1.2x'

	if dataset_str in ['WHAS']:
		hyp['n_quantiles'] = 6
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 100
			hyp['is_dropout'] = False
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = False
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['SUPPORT']:
		hyp['n_quantiles'] = 6
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = False
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['GBSG']:
		hyp['n_quantiles'] = 6
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['METABRICv2']:
		hyp['n_quantiles'] = 6
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['TMBImmuno']:
		hyp['n_quantiles'] = 6
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = False
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 100
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['BreastMSK']:
		hyp['n_quantiles'] = 6
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 100
			hyp['is_dropout'] = False
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = False
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['LGGGBM']:
		hyp['n_quantiles'] = 6
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = True
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['Housing']:
		hyp['n_quantiles'] = 20
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = False
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 100
			hyp['is_dropout'] = False
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['Protein']:
		hyp['n_quantiles'] = 20
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = False
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = False
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 100
			hyp['is_dropout'] = False
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 100
			hyp['is_dropout'] = False
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['Wine']:
		hyp['n_quantiles'] = 20
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = False
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = False
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 100
			hyp['is_dropout'] = False
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['PHM']:
		hyp['n_quantiles'] = 20
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = False
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = False
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = False
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['SurvMNISTv2']:
		hyp['n_quantiles'] = 20
		hyp['model_str'] = 'cnn_small'
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
			hyp['y_max'] = '1.2x'
		elif loss_str == 'excl_censor':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass

	# linear model likes larger lrate 0.1, NN likes 0.01, cnn likes 0.001
	if 'linear' in hyp['model_str']:
		hyp['learning_rate'] = 0.1 
	if 'cnn_small' in hyp['model_str']:
		hyp['learning_rate'] = 0.001 

	hyp['n_show'] = np.minimum(1000,hyp['n_data']) # how many datapoints to show on plots (slow to load points otherwise)
	return hyp


