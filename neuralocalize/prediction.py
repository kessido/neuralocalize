"""This code simulates the prediction code of the of the connectivity model.
"""
import gzip
import pickle
import uuid

import numpy as np
import scipy.linalg as sl
import sklearn.preprocessing

from .utils import constants
from . import feature_extraction, utils


class FeatureExtractor:
	"""A class warping the scaling and feature extraction methods.
	"""

	def __init__(self, pca_result = None, default_brain_map = None):
		""" Init the Feature Extractor from subjects and pca.
		Create a scaling factor of the cortical and sub cortical parts.

		:param pca_result: the PCA to use.
		"""
		self._is_fitted = False
		self._semi_dense_connectome_data = None
		self._left_right_hemisphere_data = None
		self._uuid = uuid.uuid4()
		self._pca_result = pca_result
		self._default_brain_map = default_brain_map
		self._ctx_indices, self._sub_ctx_indices = utils.cifti_utils.get_cortex_and_sub_cortex_indices(default_brain_map)

	def _add_features_to_subjects(self, subjects, features):
		for subject, feature in zip(subjects, features):
			subject.features_extractor_uuid = self._uuid
			subject.features_before_scaling = feature.copy()

	def _get_features_for_scaling_ctx_sub_ctx(self, subjects_features):
		subjects_features = np.transpose(subjects_features, [1, 0, 2])
		ctx_features = subjects_features[self._ctx_indices, :, :]
		sub_ctx_features = subjects_features[self._sub_ctx_indices, :, :]
		return ctx_features, sub_ctx_features

	def _set_features_for_scaling_ctx_sub_ctx(self, subjects_features, ctx_features, sub_ctx_features):
		subjects_features = np.transpose(subjects_features, [1, 0, 2])
		subjects_features[self._ctx_indices, :, :] = ctx_features
		subjects_features[self._sub_ctx_indices, :, :] = sub_ctx_features
		return np.transpose(subjects_features, [1, 0, 2])

	def _scale_transform(self, subjects_features):
		ctx_features, sub_ctx_features = self._get_features_for_scaling_ctx_sub_ctx(subjects_features)
		ctx_features = utils.utils.fsl_normalize(ctx_features)
		sub_ctx_features = utils.utils.fsl_normalize(sub_ctx_features)
		return self._set_features_for_scaling_ctx_sub_ctx(subjects_features, ctx_features, sub_ctx_features)

	def _get_or_create_semi_dense_connectome_data(self):
		if self._semi_dense_connectome_data is None:
			if self._pca_result is None:
				res, _ = utils.cifti_utils.load_cifti_brain_data_from_file(constants.DEFAULT_STRUCTURE_ICA_RESULT_PATH)
			else:
				res =feature_extraction.get_subcortical_parcellation(
					self._pca_result, self._default_brain_map)
			self._semi_dense_connectome_data = res.transpose()
		return self._semi_dense_connectome_data

	def _get_or_create_left_right_hemisphere_data(self):
		if self._left_right_hemisphere_data is None:
			if self._pca_result is None:
				res, _ = utils.cifti_utils.load_cifti_brain_data_from_file(constants.DEFAULT_ICA_SEPERATED_RESULT_PATH)
			else:
				res = feature_extraction.run_group_ica_separately(
					self._pca_result, self._default_brain_map)
			self._left_right_hemisphere_data = res.transpose()
		return self._left_right_hemisphere_data


	def _load_cached_subjects_features(self, subjects):
		res = []
		subjects_not_loaded_indices = []
		for i, subject in enumerate(subjects):
			if subject.features_extractor_uuid == self._uuid:
				res.append(subject.features_before_scaling.copy())
			else:
				res.append(None)
				subjects_not_loaded_indices.append(i)
		return res, subjects_not_loaded_indices

	def transform(self, subjects):
		"""Extract the subject features.

		:param subjects: The subjects to extract their features [n_subjects, n_data].
		:return: The subjects' features.
		"""
		print("Extracting features.")
		res, subjects_not_loaded_indices = self._load_cached_subjects_features(subjects)
		if len(subjects_not_loaded_indices) > 0:
			left_right_hemisphere_data = self._get_or_create_left_right_hemisphere_data()
			semi_dense_connectome_data = self._get_or_create_semi_dense_connectome_data()
			self._pca_result = None
			
			feature_extraction.run_dual_regression(left_right_hemisphere_data, self._default_brain_map, subjects)
			subjects_not_loaded = [subjects[i] for i in subjects_not_loaded_indices]
			feature_extraction.get_semi_dense_connectome(semi_dense_connectome_data, subjects_not_loaded)
			feature_extraction_res = [sub.correlation_coefficient.transpose() for sub in subjects_not_loaded]

			for i, subject_result in zip(subjects_not_loaded_indices, feature_extraction_res):
				res[i] = subject_result
		res = np.array(res, dtype=constants.DTYPE)
		self._add_features_to_subjects(subjects, res)
		res = self._scale_transform(res)

		return res


class Predictor:
	"""A class containing all the localizer predictor model data.

		This allow injecting another model instead, as it uses fit(x,y) and predict(x).
	"""
	
	class _DefultPredictorGenerator:
		"""Implement a standart predictor generator"""
		
		class _DefultPredictorModel:
			"""Implement a standart predictor"""
			
			def __init__(self, beta):
				self._beta = beta
			
			def predict(self, X):
				return np.array([utils.utils.add_ones_column_to_matrix(x) for x in X]) @ self._beta

		def fit(self, X, y):
			betas = []
			for subject_feature, task in zip(X, y):
				x = utils.utils.add_ones_column_to_matrix(subject_feature)
				res = sl.lstsq(x, task)[0]
				betas.append(res)
			beta = np.mean(np.array(betas), axis=0)
			return Predictor._DefultPredictorGenerator._DefultPredictorModel(beta)
			
	def __init__(self, pca_result, default_brain_map, predictor_generator=None):
		"""Init the predictor.
		"""
		self._is_fitted = False
		self._betas = None
		self._pca_result = pca_result
		self._default_brain_map = default_brain_map
		self._spatial_filters = None
		if predictor_generator is None:
			predictor_generator = Predictor._DefultPredictorGenerator()
		self._predictor_generator = predictor_generator

	def fit(self, subjects_feature, subjects_task):
		"""Fit the model from the data.

		:param subjects_feature: X,
				[n_samples, n_features] Matrix like object containing the subject features.
		:param subjects_task: y,
				[n_samples, n_results] Matrix like object containing the subject task results.
		"""
		if self._spatial_filters is None:
			if self._pca_result is None:
				group_ica_together = utils.cifti_utils.load_cifti_brain_data_from_file(constants.DEFAULT_ICA_BOTH_RESULT_PATH)[0].transpose()
			else:
				group_ica_together = feature_extraction.run_group_ica_together(self._pca_result, self._default_brain_map)
			self._pca_result = None
			self._default_brain_map = None
			self._spatial_filters = feature_extraction.get_spatial_filters(group_ica_together)
		self._predictors = []
		
		subjects_feature = utils.utils.fsl_normalize(subjects_feature)
		for j in range(self._spatial_filters.shape[1]):
			ind = self._spatial_filters[:, j] > 0
			if np.any(ind):			
				partial_subjects_features = utils.utils.fsl_demean(subjects_feature[:, ind], 1)
				partial_task = subjects_task[:,ind]
				self._predictors.append(self._predictor_generator.fit(partial_subjects_features, partial_task))
			else:
				self._predictors.append(None)
		self._is_fitted = True

	def predict(self, subjects_features):
		"""Predict the task results from the subjects features.

		:param subjects_features: X,
					[n_subjects, n_features] Matrix like object containing the subjects features.
		:return: y,
					[n_subjects, n_results] Matrix like object containing the task result prediction.
		"""
		if not self._is_fitted:
			raise BrokenPipeError("Cannot predict before the model was trained!")
		res = np.zeros((subjects_features.shape[0], self._spatial_filters.shape[0]))
			
		for j, predicator in zip(range(self._spatial_filters.shape[1]), self._predictors):
			ind = self._spatial_filters[:, j] > 0
			if np.any(ind):			
				partial_subjects_features = utils.utils.fsl_demean(subjects_features[:, ind], 1)
				res[:,ind] = predicator.predict(partial_subjects_features)
		return res

class Localizer:
	"""A class containing the localizer model data.
	"""

	def __init__(self, subjects=None, pca_result=None, compute_pca = False, number_of_pca_component = 1000, predictor_generator = None,
				 sample_file_path=constants.EXAMPLE_FILE_PATH):
		"""Initialize a localizer object
		"""
		_, brain_maps = utils.cifti_utils.load_cifti_brain_data_from_file(sample_file_path)
		
		if compute_pca and subjects is None:
			raise ValueError("Cannot run pca if no subjects were provided.")
		
		if compute_pca:
			pca_result = Localizer._get_pca(subjects, number_of_pca_component) # iterative_pca.iterative_pca(subjects)

		self._feature_extractor = FeatureExtractor(pca_result=pca_result, default_brain_map=brain_maps)												 
		self._predictor = Predictor(pca_result=pca_result, default_brain_map=brain_maps, predictor_generator=predictor_generator)

	@staticmethod
	def _get_pca(subjects, number_of_pca_component):
		incPCA = sklearn.decomposition.IncrementalPCA(number_of_pca_component)
		for subject in subjects:
			ses = np.concatenate([ses.cifti.transpose() for ses in subject.sessions], axis=1)
			incPCA.partial_fit(ses)
		print(incPCA.n_components.shape)
		return incPCA.n_components

	def fit(self, subjects, subjects_task):
		"""Fit the current loaded model on the given data.
		"""
		subjects_feature = self._feature_extractor.transform(subjects)
		self._predictor.fit(subjects_feature, subjects_task)

	def predict(self, subjects):
		"""Predict the task results from the subjects features.
		:return: The task result prediction.
		"""
		features = self._feature_extractor.transform(subjects)
		return self._predictor.predict(features)

	def save_to_file(self, file_path):
		"""Save localizer to file.
		"""
		print("Saving model to", file_path)
		return pickle.dump(self, gzip.open(file_path, 'wb'))

	@staticmethod
	def load_from_file(file_path):
		"""Load a localizer from file.
		:return: The localizer object loaded.
		"""
		print("Loading model from", file_path)
		res = pickle.load(gzip.open(file_path, 'rb'))
		if not isinstance(res, Localizer):
			raise TypeError("Content of file is either an old type and deprecated Localizer model, "
							"a corrupted file or in a wrong file format.")
		return res
