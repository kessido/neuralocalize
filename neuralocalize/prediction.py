"""This code simulates the prediction code of the of the connectivity model.
"""
import gzip
import pickle
import uuid

import numpy as np
import sklearn.preprocessing

import constants
import feature_extraction
import utils.cifti_utils
import utils.utils


class FeatureExtractor:
	"""A class warping the scaling and feature extraction methods.
	"""

	def __init__(self, pca_result,
				 is_load_features_from_file=False,
				 load_ica_result=False,
				 file_path_template=None,
				 sample_file_path=r'..\resources\example.dtseries.nii'):
		""" Init the Feature Extractor from subjects and pca.
		Create a scaling factor of the cortical and sub cortical parts.

		:param pca_result: the PCA to use.
		:param is_load_features_from_file: If True, load the features from file using the file path template.
		:param file_path_template: The template to load the features from.
				Usage: file_path_template % subject.name
		"""
		self._is_fitted = False
		self._should_load_features_from_file = is_load_features_from_file
		self._load_ica_result = load_ica_result
		self._file_path_template = file_path_template
		self._semi_dense_connectome_data = None
		self._left_right_hemisphere_data = None
		self._uuid = uuid.uuid4()
		self._pca_result = pca_result
		_, self._default_brain_map = utils.cifti_utils.load_cifti_brain_data_from_file(sample_file_path)
		self._ctx_indices, self._sub_ctx_indices = utils.cifti_utils.get_cortex_and_sub_cortex_indices(sample_file_path)

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
			self._semi_dense_connectome_data = feature_extraction.get_subcortical_parcellation(
				self._pca_result, self._default_brain_map, load_ica_result=self._load_ica_result)
		return self._semi_dense_connectome_data

	def _get_or_create_left_right_hemisphere_data(self):
		if self._left_right_hemisphere_data is None:
			self._left_right_hemisphere_data = feature_extraction.run_group_ica_separately(
				self._pca_result, self._default_brain_map, load_ica_result=self._load_ica_result).transpose()
		return self._left_right_hemisphere_data

	def _load_features_from_file(self, subjects):
		"""Load the subject features from file.

		:param subjects:
		:return:
		"""
		res = []
		for subject in subjects:
			path = self._file_path_template % subject.name
			img, _ = utils.cifti_utils.load_cifti_brain_data_from_file(path)
			res.append(img.transpose())
		return res

	def _load_cached_subjects_features(self, subjects):
		"""Load the subjects features from the cached features.

		:param subjects: The subject to load
		:return:
			res: List of the result features of each subject.
				 In place where the subjects features could not be loaded from cache, place None.
			subjects_not_loaded_indices: List of indices where it was not possible to load the subjects.
		"""
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
		if self._should_load_features_from_file:
			res = self._load_features_from_file(subjects)
		else:
			res, subjects_not_loaded_indices = self._load_cached_subjects_features(subjects)
			if len(subjects_not_loaded_indices) > 0:
				feature_extraction.run_dual_regression(self._get_or_create_left_right_hemisphere_data(),
													   self._default_brain_map, subjects)

				semi_dense_connectome_data = self._get_or_create_semi_dense_connectome_data().transpose()
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

	def __init__(self, pca_result, brain_maps=None, load_ica_result=False,
				 sample_file_path=r'..\resources\example.dtseries.nii'):
		"""Init the predictor.

		:param pca_result: The pca to extract the spatial filtering from.
						This is later user to group indexes by their connectivity ICA,
						and combine them as their group only predictors.
		"""
		self._is_fitted = False
		self._betas = None
		if brain_maps is None:
			_, self._brain_maps = utils.cifti_utils.load_cifti_brain_data_from_file(sample_file_path)
		self._pca_result = pca_result
		self._load_ica_result = load_ica_result
		self._spatial_filters = None

	def _get_beta(self, subject_features, subject_task):
		"""Get the prediction betas from psudo-inverse of ((beta @ [1 subject_features] = subject_task)).

		:param subject_features: The subject features.
		:param subject_task: The subject task results.
		:return: The subject betas.
		"""
		task = subject_task
		# TODO(loya) do we get this before or after the transposition?
		subject_features = utils.utils.fsl_normalize(subject_features)
		betas = np.zeros(
			(subject_features.shape[1] + 1, self._spatial_filters.shape[1]))
		for j in range(self._spatial_filters.shape[1]):
			ind = self._spatial_filters[:, j] > 0
			if np.any(ind):
				y = task[ind]
				demeaned_features = utils.utils.fsl_demean(subject_features[ind])
				x = utils.utils.add_ones_column_to_matrix(demeaned_features)
				betas[:, j] = np.linalg.pinv(x) @ y
		return betas

	def fit(self, subjects_feature, subjects_task):
		"""Fit the model from the data.

		:param subjects_feature: X,
				[n_samples, n_features] Matrix like object containing the subject features.
		:param subjects_task: y,
				[n_samples, n_results] Matrix like object containing the subject task results.
		"""
		if self._spatial_filters is None:
			self._spatial_filters = feature_extraction.get_spatial_filters(
				self._pca_result, self._brain_maps, self._load_ica_result)
		betas = []
		for subject_feature, task in zip(subjects_feature, subjects_task):
			betas.append(self._get_beta(subject_feature, task))
		betas = np.array(betas, dtype=constants.DTYPE)
		self._betas = betas
		self._is_fitted = True

	def _predict(self, subject_features):
		res = np.zeros(self._spatial_filters.shape[0])
		betas_after_transpose = self._betas.transpose([1, 2, 0])  # TODO(loya) added the transposition to the scaling
		for j in range(self._spatial_filters.shape[1]):
			ind = self._spatial_filters[:, j] > 0
			if np.any(ind):
				demeaned_features = sklearn.preprocessing.scale(subject_features[ind], with_std=False)
				x = utils.utils.add_ones_column_to_matrix(demeaned_features)
				current_betas = betas_after_transpose[:, j, :]
				res[ind] = x @ np.mean(current_betas, axis=1)
		return res

	def predict(self, subjects_features):
		"""Predict the task results from the subjects features.

		:param subjects_features: X,
					[n_subjects, n_features] Matrix like object containing the subjects features.
		:return: y,
					[n_subjects, n_results] Matrix like object containing the task result prediction.
		"""
		if not self._is_fitted:
			raise BrokenPipeError("Cannot predict before the model was trained!")
		return np.array([self._predict(subject_features) for subject_features in subjects_features],
						dtype=constants.DTYPE)


class Localizer:
	"""A class containing the localizer model data.
	"""

	def __init__(self, subjects=None, pca_result=None,
				 load_subjects_features_from_file=False,
				 subjects_feature_path_template=None):
		"""Initialize a localizer object

		:param subjects: The subject to compute the pca_result from.
		:param pca_result: The pca to use for the features extraction and ICA filtering.
					If not provided, you must provide subjects to create the PCA from.
		:param load_subjects_features_from_file: If True, load the features from file instead.
		:param subjects_feature_path_template: If load_subjects_features_from_file is True, use 
					this file template 
		"""
		if pca_result is None and not subjects:
			raise ValueError("Cannot initialize a localizer if no pca and no subjects were provided, " +
							 "as it cannot generate a new PCA without subjects.")
		
		if load_subjects_features_from_file and subjects_feature_path_template is None:
			raise ValueError("If you with to load the subject features from path, you must provide a template path to load with.")
		
		if pca_result is None:
			pca_result = Localizer._get_pca(subjects)  # iterative_pca.iterative_pca(subjects)

		self._feature_extractor = FeatureExtractor(pca_result=pca_result,
												 load_subjects_features_from_file=load_subjects_features_from_file,
												 subjects_feature_path_template=subjects_feature_path_template)
												 
		self._predictor = Predictor(pca_result)

	@staticmethod
	def _get_pca(subjects):
		return sklearn.decomposition.IncrementalPCA(1000, whiten=True).fit(
			np.concatenate(
				[np.concatenate([ses.cifti.transpose() for ses in subject.sessions], axis=0) for subject in subjects],
				axis=0))

	def fit(self, subjects, subjects_task):
		"""Fit the current loaded model on the given data.

		:param subjects: The subject to fit on.
		:param subjects_task: The task result of each subject.
		:return:
		"""
		subjects_feature = self._feature_extractor.transform(subjects)
		self._predictor.fit(subjects_feature, subjects_task)

	def predict(self, subjects):
		"""Predict the task results from the subjects features.

		:param subjects: The subjects to predict his task results.
		:return: The task result prediction.
		"""
		features = self._feature_extractor.transform(subjects)
		return self._predictor.predict(features)

	def save_to_file(self, file_path):
		"""Save localizer to file.

		:param file_path: Path to save the object to.
		"""
		print("Saving model to", file_path)
		return pickle.dump(self, gzip.open(file_path, 'wb'))

	@staticmethod
	def load_from_file(file_path):
		"""Load a localizer from file.

		:param file_path: File path to load from.
		:return: The localizer object loaded.
		"""
		print("Loading model from", file_path)
		res = pickle.load(gzip.open(file_path, 'rb'))
		if not isinstance(res, Localizer):
			raise TypeError("Content of file is either an old type and deprecated Localizer model, "
							"a corrupted file or in a wrong file format.")
		return res
