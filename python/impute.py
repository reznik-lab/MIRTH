import os
import numpy as np
import pandas as pd
import argparse
import csv
import warnings
from autograd_minimize import minimize
import torch


#### COMMAND-LINE ARGS ####
parser = argparse.ArgumentParser(description="Impute metabolomics data with MIRTH (Metabolite Imputation by Rank Transformation and Harmonization)",
	formatter_class=argparse.RawTextHelpFormatter)

# Options for input
parser.add_argument('-in', '--input_dir', help='Directory where raw data to be imputed is stored', required=False, default='data', type=str)
# Options for NMF
parser.add_argument('-d', '--ndims', help='Number of embedding dimensions for NMF', required=False, default=30, type=int)
parser.add_argument('-s', '--seed', help="Set seed", required=False, default=42, type=int)
# Options for cross-validation
parser.add_argument('-c', '--crossval', help='Enable cross-validation. Overrides number of dimensions (-d) and greatly increases runtime.', required=False, action='store_true', default=False)
parser.add_argument('-cd', '--crossval_dims', help='Range of embedding dimensions to evaluate in cross-validation', required=False, nargs=2, type=int, default=[20, 30])
parser.add_argument('-cf', '--crossval_folds', help='Number of folds for cross-validation', required=False, type=int, default=10)
# Options for output
parser.add_argument('-rd', '--results_dir', help='Optionally specify the name of the results directory', required=False, type=str, default='MIRTH_out')

args = parser.parse_args()

#### SETUP ####

# Initialize variables from command-line arguments
raw_data_dir = args.input_dir
n_dims = args.ndims
seed = args.seed
cv = args.crossval
cv_dims = args.crossval_dims
cv_folds = args.crossval_folds
results_dir = args.results_dir


# if results directory doesn't exits, create it
try:
	os.mkdir(results_dir)
except OSError as error:
	warnings.warn(f'The directory {results_dir} already exists. Results may be overwritten!')

# Override n_dims assignment when cross-validation is enabled
if cv:
	print('Cross-validation enabled')
	n_dims = None

	# initialize CSV file to store error/scores at each fold and for each pair of regularization parameters
	out_header = ["n_dims","fold","mae"]
	with open(f'{results_dir}/cv_folds_scores.csv', 'w') as file:
		writer = csv.writer(file)
		writer.writerow(out_header)


#### FUNCTION DEFINITIONS ####

def load_data():
	features = set()
	nrows = 0
	samples = [] 
	batch_names = []

	# reads sample and feature names of the data to get maps and size
	for batch_idx, fpath in enumerate(os.listdir(raw_data_dir)):
		if 'csv' not in fpath:
			continue
		print(f'Reading {raw_data_dir}/{fpath}')
		df = pd.read_csv(f'{raw_data_dir}/{fpath}', header=0, index_col=0).T

		features.update(df.columns)
		samples.extend(df.index)
		batch_names.append(fpath)
		nrows += df.shape[0]

	print(f'{len(features)} total unique features.')
	print(f'{nrows} total samples.')

	# create map of features and samples
	feature_map = {s:i for i,s in enumerate(features)}
	sample_map = {s:i for i,s in enumerate(samples)}
	batch_map = {s:i for i,s in enumerate(batch_names)}

	data = np.full((nrows, len(features)), np.nan)
	# a vector that gives the batch index at every row of the raw data matrix (identifies which batch a given sample comes from based on its row index)
	# formerly known as "batches"
	batch_index_vector = np.zeros(nrows, dtype=int)

	# read raw data and assemble data matrix
	sidx = 0
	batch_idx = 0
	for fpath in os.listdir(f'{raw_data_dir}'):
		if 'csv' not in fpath:
			continue
		df = pd.read_csv(f'{raw_data_dir}/{fpath}', header=0, index_col=0).T

		# Add the data to the measured columns
		for feature in df.columns:
			fidx = feature_map[feature]
			data[sidx:sidx+df.shape[0], fidx] = df[feature].values
			# sample_map

		# Track which batch each sample belongs to
		batch_index_vector[sidx:sidx+df.shape[0]] = batch_idx

		# Update the sample index
		sidx += df.shape[0]

		# Update the batch index
		batch_idx += 1


	return data, batch_index_vector, feature_map, sample_map, batch_map

def convert_to_ranks(data, batch_index_vector):

	n_batches = batch_index_vector.max()+1

	for bidx in range(n_batches):
		# Get the batch
		batch_rows = np.arange(data.shape[0])[batch_index_vector == bidx]
		# subset for just one batch
		batch = data[batch_rows]

		# Features where there is no data will be ignored in the subsequent rank conversion step
		missing = np.all(np.isnan(batch), axis=0)

		for col in range(data.shape[1]):
			if missing[col]:
				continue

			# Find the rows where there is non-missing data in the current column
			missing_in_col = np.isnan(batch[:,col])
			rows = batch_rows[~missing_in_col]
			# extract those non-missing values
			vals = data[rows, col]

			# Normalize by the length
			normalizer = len(batch)+1

			# Convert to ranks
			order = vals.argsort()
			data[rows[order], col] = np.arange(missing_in_col.sum()+1,1+len(batch)) / normalizer

			# For any missing values, they are all tied at the bottom rank, halfway between the minimum possible rank and the rest of the ranks of non-missing values
			data[batch_rows[missing_in_col], col] = (missing_in_col.sum()+1) * 0.5 / normalizer

	return data

def tic_normalization(data,batch_index_vector):

	# This function will be used as the simplest case of normalization to see
	# what effect it has on the data. We're getting better results without
	# normalization now, which might be caused by bias in the data and would not
	# generalize well. For this normalization, present values are divided by the
	#  sum of the row and left-censored data considered as 1/2 the lowest
	# measured value in the batch, but is not filled in until the rank-conversion occurs

	print('Normalizing data...')

	# Make a copy of the data so we can compare normalized to unnormalized
	normalized_data = np.copy(data)

	# Row normalize one batch at a time
	n_batches = batch_index_vector.max()+1
	for bidx in range(n_batches):
		# Get the batch
		batch_rows = np.arange(data.shape[0])[batch_index_vector == bidx]
		# Subset data to get just one batch
		batch = data[batch_rows]
	
		# need to keep track of where all the censored (and missing) data is
		nan_mask = np.isnan(batch)

		# where are the entirely missing columns?
		missing = np.all(nan_mask, axis=0)

		# minimum value in the batch
		min_batch = np.nanmin(batch)
		
		# loop through each row in the batch
		for row in range(batch.shape[0]):

			# at one row, count all the places with missing values.
			# subtract the number of missing values we know are there due to entirely missing features
			n_censored = np.sum(nan_mask[row])-np.sum(missing)
			# total ion count for the row, counting the many censored data points
			row_tic = np.nansum(batch[row]) + 0.5 * n_censored * min_batch
			
			# replace batch data	
			batch[row,:] = batch[row,:]/row_tic

		# Replace present data with the value divided by the sum of the row
		batch = batch / np.nansum(batch, axis=1, keepdims=True)
		# Put the batch back in normalized_data so you can move on to the next batch

		normalized_data[batch_rows] = batch

	return normalized_data

def split_folds(data, batch_index_vector, n_folds):

	# This function splits the data into folds for cross-validation.
	# If we are running the nmf on only a single batch, half the samples are
	# used for cross valdidation (so the other half can be used for testing).
	# Otherwise, all samples could be used since there will be another batch
	# to test on.
	# The code isolates a batch, determines features available to use, splits up
	# the available features into folds (as many as are specified by n_folds),
	# then selects samples to complete the fold.
	# is that right??

	n_batches = batch_index_vector.max()+1
	folds = [[] for _ in range(n_folds)]

	# In the single-batch case, only a subset of samples should be held out for
	# cross-validation. For multi-batch case, the 'available' condition ensures that
	# if a feature is chosen in a batch, there will be at least one other batch
	# with data on that feature which can be used for testing (aka the solo
	# condition takes care of avoiding non-overlapping features).

	if n_batches.max() < 3:
		cv_sample_prop = 0.5
	else:
		cv_sample_prop = 1

	# Different features are held out for cross-validation in each batch.
	for bidx in range(n_batches):
		# Subset the data for the batch
		batch = data[batch_index_vector == bidx]

		# Get features where there is ANY missing data (will exclude these)
		missing = np.any(np.isnan(batch), axis=0)

		# Get features where there is no other batch with this feature (will exclude these too)
		solo = np.all(np.isnan(data[batch_index_vector != bidx]), axis=0)
		
		# in single-set, override solo condition (no other features will be present in other batches because there are not other batches)
		if n_batches.max() == 1:
			solo = ~solo

		# Get the list of available features to split into folds
		# in double-set case, [(~missing) & (~solo)] will be the same for both sets. (missing(set1)=solo(set2), solo(set1)=missing(set2))
		# so the feature split will be the same every time
		available = np.arange(batch.shape[1])[(~missing) & (~solo)]
		
		# Reorder available to decrease feature overlap between folds
		np.random.shuffle(available)
		
		# Split the array into equal parts, (should we be randomly scrambling features between folds in every batch)

		splits = np.array_split(available, n_folds)

		# Add the split array (an array of arrays of column indices) to the folds array

		rows = np.arange(data.shape[0])[batch_index_vector == bidx]

		# (In each batch,) hold out a random set of samples in each fold
		for fold_idx in range(n_folds):
			cols = splits[fold_idx]

			for cidx in cols:

				random_rows = np.random.choice(rows,size=round(batch.shape[0]*cv_sample_prop),replace=False)

			# is that done ---> ??? # Eventually, randomize the choice of row for each feature in a fold
				folds[fold_idx].extend([[x, cidx] for x in random_rows])

	# Convert each fold to numpy arrays
	folds = [np.array(fold) for fold in folds]

	# print(folds)

	return folds

def cross_validation(training, batch_index_vector, n_folds):
	# This function executes cross validation on the training data to determine
	# the optimal number of embedding dimensions...

	print('Beginning cross-validation (this will take a while)')

	# Split data into folds
	folds = split_folds(training, batch_index_vector,n_folds)

	n_dims_options = np.arange(start=cv_dims[0], stop=cv_dims[1])

	# Initialize array of scores
	scores = np.zeros((len(n_dims_options), n_folds))

	for didx, n_dims in enumerate(n_dims_options):

		print(f'n_dims: {n_dims}')

		for fidx, fold in enumerate(folds):
			# Mask the training data
			cv_training = np.copy(training)
			
			cv_training[fold[:,0], fold[:,1]] = np.nan

			W, H = nmf(X=cv_training, n_dims=n_dims)
			# W is the coefficients (samples) matrix
			# H is the features matrix

			# X is reconstructed matrix from multiplication of approximate factors of data
			X = W.dot(H)

			# Convert matrix to batch ranks
			X_ranks = convert_to_ranks(X,batch_index_vector)

			scores[didx, fidx] = np.abs(X_ranks[fold[:,0], fold[:,1]] - training[fold[:,0], fold[:,1]]).mean()

			print(f'\tfold {fidx}: {scores[didx, fidx]:.2f}')

			# write scores to CSV file
			with open(f'{results_dir}/cv_folds_scores.csv','a') as file:
				writer = csv.writer(file)
				writer.writerow([n_dims,fidx,scores[didx, fidx]])

	# Get the best scoring number of dimensions
	n_dims = n_dims_options[np.argmin(scores.mean(axis=1))]

	print(f'Best dims is {n_dims}')

	return n_dims

## NMF FUNCTION
def nmf(X, n_dims):
    n_samples, n_features = X.shape

    # create initial input, which is dict with random starting embedding matrices
    x0 = {'W': np.random.random(size = (n_samples,n_dims)), 'H': np.random.random(size = (n_dims,n_features))}

    t_X = torch.Tensor(X)

    # define loss function
    def nmf_loss(W=None, H=None):

    	# skipping over NaNs
        out = torch.matmul(W,H)
        nan = torch.isnan(t_X)
        y = torch.where(nan, torch.tensor(0.0), t_X)
        out = torch.where(nan, torch.tensor(0.0), out)
        L = torch.sum((out - y) ** 2)
        
        return L

    res = minimize(nmf_loss,x0, method='L-BFGS-B', bounds=(0, None), backend='torch')

    W = res.x['W']
    H = res.x['H']
    
    if res.success:
       print('Minimization successful!')
    else:
        print('Minimization failed! Interpret with caution.')
        print('Reason failed:')
        print(f'\t{res.message}')

    return W, H

def save_embedding(matrix,mapping,mode):
	# W is coefficients (samples) matrix, columns are dims (rows are samples)
	# H is the features matrix, with columns as features
	if mode == "features":
		features_from_index = {v: k for k, v in mapping.items()}
		feature_names = [features_from_index[i] for i in range(len(mapping))]
		# Now write everything to file
		#
		#dim_lab_template = np.full(matrix.shape[0],'dimensions')
		dim_lab_num = np.arange(start=1,stop=matrix.shape[0]+1)#.astype('|S2')
		#dim_labs = np.core.defchararray.add(dim_lab_template,dim_lab_num)
		pd.DataFrame(matrix, index=dim_lab_num, columns=feature_names).to_csv(f'{results_dir}/feature_embedding_H.csv')
	elif mode == "samples":
		samples_from_index = {v: k for k, v in mapping.items()}
		sample_names = [samples_from_index[i] for i in range(len(mapping))]

		dim_lab_num = np.arange(start=1,stop=matrix.shape[1]+1)
		pd.DataFrame(matrix, index=sample_names, columns=dim_lab_num).to_csv(f'{results_dir}/sample_embedding_W.csv')

	print(f'Saved {mode} embedding to directory {results_dir}.')


#### IMPUTATION ####

# set seed
np.random.seed(seed)

# LOAD DATA
data, batch_index_vector, feature_map, sample_map, batch_map = load_data()

# get feature and sample names from respective maps
features_from_index = {v: k for k, v in feature_map.items()}
feature_names = [features_from_index[i] for i in range(len(feature_map))]

samples_from_index = {v: k for k, v in sample_map.items()}
sample_names = [samples_from_index[i] for i in range(len(sample_map))]

# Can save copies of feature_map, sample_map, and batch_map for debug
#pd.DataFrame(feature_map.items(), columns=['feature_name','feature_index']).to_csv(f'{results_dir}/feature_map.csv')
#pd.DataFrame(sample_map.items(), columns=['sample_name','sample_index']).to_csv(f'{results_dir}/sample_map.csv')
#pd.DataFrame(batch_map.items(), columns=['batch_name','batch_index']).to_csv(f'{results_dir}/batch_map.csv')

# Save aggregate raw data
pd.DataFrame(data, index=sample_names, columns=feature_names).to_csv(f'{results_dir}/raw_aggregate_data.csv')

# TIC-normalize data
data = tic_normalization(data,batch_index_vector)

# Save TIC-normalized data
pd.DataFrame(data, index=sample_names, columns=feature_names).to_csv(f'{results_dir}/normalized_aggregate_data.csv')

# Convert TIC-normalized data to ranks
ranked_data = convert_to_ranks(data, batch_index_vector)
pd.DataFrame(ranked_data, index=sample_names, columns=feature_names).to_csv(f'{results_dir}/ranked_aggregate_data.csv')

# Run cross-validation if enabled:
if n_dims is None:
		n_dims = cross_validation(ranked_data, batch_index_vector, cv_folds)

# factorize ranked data with best n_dims and reconstruct matrix
print('Imputing data...')
W, H = nmf(X=ranked_data, n_dims=n_dims)
X_predictions = W@H

# remap to ranks
ranked_predictions = convert_to_ranks(X_predictions,batch_index_vector)

# save copies of samples and features embedding
save_embedding(H,feature_map,"features")
save_embedding(W,sample_map,"samples")

# save imputed data
pd.DataFrame(ranked_predictions, index=sample_names, columns=feature_names).to_csv(f'{results_dir}/imputed_aggregate_data.csv')


