"""This code simulates the feature extraction part of the connectivity model.
"""
import numpy as np
import numpy.matlib as matlib
import scipy.signal
import scipy.linalg as sl
import sklearn
import sklearn.decomposition

from .utils import constants 
from . import utils
from .utils import utils as util
from .utils.constants import ICA_THRESHOLD_CONST, DTYPE


def ica_with_threshold(image, num_ic, threshold):
	"""Run ica and threshold and parcellate the results.
	"""
	ica = sklearn.decomposition.FastICA(n_components=num_ic)
	ica_result = ica.fit_transform(image.transpose()).transpose()  # Reconstruct signals
	thresh = (np.abs(ica_result) > threshold).astype(DTYPE)
	ica_time_thresh = ica_result * thresh
	end_res = np.sign(np.sum(np.sign(ica_time_thresh), 1))
	end_res_t = np.reshape(end_res, (num_ic, 1))
	tile_res = np.tile(end_res_t, (1, ica_result.shape[1]))
	return ica_result * tile_res


def run_group_ica_separately(cifti_image, BM, threshold=ICA_THRESHOLD_CONST, num_ic=40, N=91282):
	"""Runs a group ICA for each hemisphere separately
	"""
	print("Running Group ICA on each hemisphere separately.")
	# TODO (Itay) cifti_image to left_hemisphere_data and right_hemisphere_data
	left_hemisphere_data = cifti_extract_data(cifti_image, BM, 'L')
	right_hemisphere_data = cifti_extract_data(cifti_image, BM, 'R')

	left_ica = ica_with_threshold(left_hemisphere_data, num_ic, threshold)
	right_ica = ica_with_threshold(right_hemisphere_data, num_ic, threshold)

	# keep ICA components that have L/R symmetry
	# left-right DICE of cortical ICs to
	# 1) re-order the ICs
	# 2) select the ICs that are found in both hemispheres
	x = np.zeros([32492, left_ica.shape[0]])
	y = np.zeros([32492, right_ica.shape[0]])
	x[BM[0].surface_indices, :x.shape[1]] = left_ica.transpose()
	y[BM[1].surface_indices, :y.shape[1]] = right_ica.transpose()

	threshold_2 = ICA_THRESHOLD_CONST

	D = dice(x > threshold_2, y > threshold_2)
	D_threshold = (D == np.transpose(np.matlib.repmat(np.amax(D, 1), D.shape[1], 1))).astype(DTYPE)
	D_tmp = ((D * D_threshold) == np.matlib.repmat(np.amax(D * D_threshold, axis=0), D.shape[1], 1))
	D_threshold = D_tmp * D_threshold

	r = np.nonzero(np.sum(D_threshold, 1))[0]
	c = D_threshold.argmax(1)
	c = c[r]

	x = np.zeros((N, len(r)))
	x[BM[0].data_indices, :x.shape[1]] = np.transpose(left_ica[r, :])
	x[BM[1].data_indices, :len(c)] = np.transpose(right_ica[c, :])
	
	return x.transpose()


def cifti_extract_data(cifti_image, BM, side):
	"""extracts data from cifti images
	"""
	if side == 'L':
		data = cifti_image[:, BM[0].data_indices]
	else:
		if side == 'R':
			data = cifti_image[:, BM[1].data_indices]
		else:
			if side == 'both':
				data = cifti_image
			else:
				raise ValueError('error: bad cifti_extract_data command, side is not L, R or both')
	return data


def dice(x, y):
	"""gets x = N*nx and y = N*ny and return nx*ny

	:param x: should be N*nx:
	:param y: should be N*ny:
	:return: nx*ny
	"""
	if x.shape[0] != y.shape[0]:
		raise ValueError('x and y incompatible (dice)')
	nx = x.shape[1]
	ny = y.shape[1]

	xx = np.matlib.repmat(np.sum(x, 0), ny, 1).transpose()
	yy = np.matlib.repmat(np.sum(y, 0), nx, 1)

	x = x.astype(DTYPE)
	y = y.astype(DTYPE)

	temp = x.transpose() @ y
	res = 2 * temp / (xx + yy)
	return res


def run_group_ica_together(pca_result, BM, threshold=ICA_THRESHOLD_CONST, num_ic=50):
	"""Runs a group ICA for both hemispheres, to use as spatial filters.
	"""
	print("Running Group ICA on both hemispheres.")
	both_hemisphere_data = cifti_extract_data(pca_result, BM, 'both')
	both_ica = ica_with_threshold(both_hemisphere_data, num_ic, threshold)
	
	return both_ica.transpose()


def run_dual_regression(left_right_hemisphere_data, BM, subjects, size_of_g=91282):
	"""Runs dual regression.

	Updates the cifti image in every subject.
	"""
	print("Running Dual Regression.")
	single_hemisphere_shape = left_right_hemisphere_data.shape[1]
	G = np.zeros([size_of_g, single_hemisphere_shape * 2])
	hemis = np.zeros([size_of_g, single_hemisphere_shape * 2])

	G[BM[0].data_indices, :single_hemisphere_shape] = left_right_hemisphere_data[BM[0].data_indices, :]
	G[BM[1].data_indices, single_hemisphere_shape: 2 * single_hemisphere_shape] = left_right_hemisphere_data[
																				  BM[1].data_indices, :]

	hemis[BM[0].data_indices, :single_hemisphere_shape] = 1
	hemis[BM[1].data_indices, single_hemisphere_shape: 2 * single_hemisphere_shape] = 1

	g_pseudo_inverse = np.linalg.pinv(G)
	for subject in subjects:
		subject_data = []
		for session in subject.sessions:
			inp = session.cifti
			normalized_cifti = utils.utils.fsl_variance_normalize(inp).transpose()
			deterended_data = np.transpose(
				scipy.signal.detrend(np.transpose(normalized_cifti), type='constant', axis=0))
			subject_data.append(deterended_data)
		subject_data = np.concatenate(subject_data, axis=1)
		T = g_pseudo_inverse @ subject_data
		t = util.fsl_glm(np.transpose(T), np.transpose(subject_data))
		subject.left_right_hemisphere_data = np.transpose(t) * hemis


def get_subcortical_parcellation(cifti_image, brain_maps):
	"""Get sub-cortical parcellation using atlas definitions and current data.
	:return: (no. voxel, cortical parcellation parts)
	"""

	def do_nothing_brain_map_handler(*args):
		pass

	def use_as_is_brain_map_handler(cifti_image, current_map):
		"""Uses the brain map with no prepossessing
		:return: numpy array (no. voxels, 1), 1 if index is in part of the current part, 0 otherwise
		"""
		ret = np.zeros([cifti_image.shape[1], 1])
		ret[current_map.data_indices] = 1
		return ret

	def corrcoef_and_spectral_ordering(mat):
		"""Implementation of reord2 + corrcoef function that was used in the matlab version
		:return: spectral ordering of the corrcoef matrix of A
		"""
		mat = np.corrcoef(mat.transpose()) + 1
		ti = np.diag(np.sqrt(1. / np.sum(mat, 0)))
		W = np.matmul(np.matmul(ti, mat), ti)
		U, S, V = np.linalg.svd(W)
		S = np.diag(S)
		P = np.multiply(np.matmul(ti, np.reshape(U[:, 1], [U.shape[0], 1])), np.tile(S[1, 1], (U.shape[0], 1)))
		return P

	def half_split_using_corrcoef_and_spectral_ordering_brain_map_handler(cifti_image, current_map):
		"""This split the data into 2 different clusters using corrcoef,
		spatial ordering, and positive\negative split
		:return: numpy array (no. voxels , 2), each vector is a 0\1 vector representing the 2 clusters
		"""
		res = np.zeros([cifti_image.shape[1], 2])
		cifti_current_map_data = cifti_image[:, current_map.data_indices]
		spatial_ordering = corrcoef_and_spectral_ordering(cifti_current_map_data)
		res[current_map.data_indices, :] = np.hstack((spatial_ordering > 0, spatial_ordering < 0)).astype(float)
		res[current_map.data_indices, :] = np.hstack((spatial_ordering > 0, spatial_ordering < 0)).astype(DTYPE)
		return res

	def label_to_function(label):
		label = label.rsplit('_', 1)[0]
		labels_to_function = {
			'CIFTI_STRUCTURE_CORTEX': do_nothing_brain_map_handler,
			'CIFTI_STRUCTURE_ACCUMBENS': use_as_is_brain_map_handler,
			'CIFTI_STRUCTURE_AMYGDALA': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
			'CIFTI_STRUCTURE_BRAIN': do_nothing_brain_map_handler,
			'CIFTI_STRUCTURE_CAUDATE': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
			'CIFTI_STRUCTURE_CEREBELLUM': ica_clustering_brain_map_handler,
			'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL': do_nothing_brain_map_handler,
			'CIFTI_STRUCTURE_HIPPOCAMPUS': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
			'CIFTI_STRUCTURE_PALLIDUM': use_as_is_brain_map_handler,
			'CIFTI_STRUCTURE_PUTAMEN': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
			'CIFTI_STRUCTURE_THALAMUS': ica_clustering_brain_map_handler,
		}
		return labels_to_function[label]

	def ica_clustering_brain_map_handler(cifti_image, current_map):
		"""This split the data into 3 parts by counting the eddect of each of the 3 first components
		in the ICA analysis on all the voxels, and determines the cluster by the one with the maximum
		connection to the voxel.
		:return: numpy array (no. voxels , 3), each vector is a 0\1 vector representing the 3 clusters
		"""
		cifti_current_map_data = cifti_image[:, current_map.data_indices]
		fastica = sklearn.decomposition.FastICA(3)
		ica_y = fastica.fit_transform(cifti_current_map_data.transpose()).transpose()

		thresh = np.asarray(np.abs(ica_y) > ICA_THRESHOLD_CONST, dtype=DTYPE)

		ica_time_thresh = ica_y * thresh
		end_res = np.sign(np.sum(np.sign(ica_time_thresh), axis=1))
		end_res_reshaped = np.reshape(end_res, (3, 1))
		end_res_t = np.matlib.repmat(end_res_reshaped, 1, ica_y.shape[1])
		ica_y = ica_y * end_res_t

		res = np.zeros([cifti_image.shape[1], ica_y.shape[0]])
		res[current_map.data_indices, :] = ica_y.transpose()
		return res

	print("Getting Subcortical parcellation.")

	sub_cortex_clusters = []
	for current_map in brain_maps:
		x = label_to_function(current_map.brain_structure_name)(cifti_image, current_map)
		if x is not None:
			sub_cortex_clusters.append(x)
	return np.hstack(sub_cortex_clusters).transpose()


def get_semi_dense_connectome(semi_dense_connectome_data, subjects):
	"""Final feature extraction (forming semi-dense connectome)
	For each subject, load RFMRI data, then load ROIs from above to calculate semi-dense connectome.

	# ASSUMES:
	# getting a subject list holding session array sessions, each holding left h. data, right h. data, ROIs, BM and CIFTI.
	# In MATLAB they're all being loaded. All these members are assumed to be numpy arrays.
	# (This is handled as an object but can be changed to a list of tuples or dictionary, whatever)

	:return: A dictionary from a subject to its correlation coeff.
	"""
	print("Running Get Semi-Dense Connectome")
	for subject in subjects:
		W = []
		ROIS = np.concatenate([subject.left_right_hemisphere_data, semi_dense_connectome_data], axis=1)
		for session in subject.sessions:
			scaled = utils.utils.fsl_demean(session.cifti)
			scaled = utils.utils.fsl_variance_normalize(scaled)
			scaled = scaled.transpose()
			W.append(scaled)
		W = np.concatenate(W, axis=1)
		# MULTIPLE REGRESSION
		T = sl.lstsq(ROIS, W)[0]
		# CORRELATION COEFFICIENT
		normalized_T = utils.utils.fsl_normalize(T, 1)
		normalized_W = utils.utils.fsl_normalize(np.transpose(W), 0)
		F = normalized_T @ normalized_W
		subject.correlation_coefficient = F


def get_spatial_filters(group_ica_together):
	"""Gets the filters (a result of the ica on the pca result), uses threshold and do winner-take-all
	The returned matrix is an index matrix which is MATLAB compatible.
	"""
	print("Getting Spatial Filters.")
	spatial_const = ICA_THRESHOLD_CONST

	m = np.amax(group_ica_together, axis=1)

	# +1 for MATLAB compatibility
	wta = np.argmax(group_ica_together, axis=1) + 1
	wta = wta * (m > spatial_const)

	S = np.zeros_like(group_ica_together)
	for i in range(group_ica_together.shape[1]):
		# +1 for MATLAB compatibility
		S[:, i] = (wta == i + 1).astype(float)
	return S
