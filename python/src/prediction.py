"""This code simulates the prediction code of the of the connectivity model.
"""

import gzip
import pickle
from constants import dtype
import numpy as np
import sklearn.preprocessing
import scipy.io
import feature_extraction
import iterative_pca
import utils.cifti_utils
import utils.utils


class FeatureExtractor:
    """A class warping the scaling and feature extraction methods.
    """

    def __init__(self, subjects, pca_result, sample_file_path='../resources/example.dtseries.nii'):
        """ Init the Feature Extractor from subjects and pca.
        Create a scaling factor of the cortical and sub cortical parts.

        :param subjects: The subjects
        :param pca_result: the PCA  to use.
        """
        self.semi_dense_connectome_data = None
        self.pca_result = pca_result

        _, self.default_brain_map = utils.cifti_utils.load_nii_brain_data_from_file(sample_file_path)

        self.ctx_indices, self.sub_ctx_indices = utils.cifti_utils.get_cortex_and_sub_cortex_indices(sample_file_path)
        # [subjects x features (tasks x brain)]
        features = self.extract(subjects, False)
        features = np.array(features)

        # TODO(kess) check if this doesn't just return the same result as scaling by the whole thing.
        # self.scaler_ctx = sklearn.preprocessing.StandardScaler().fit(features[:, self.ctx_indices])
        print("ctx shape:", self.ctx_indices.shape, "features shape:", features.shape)

        flattened_ctx_features = utils.utils.flatten_features_for_scale(features[:, self.ctx_indices])
        flattened_sub_ctx_features = utils.utils.flatten_features_for_scale(features[:, self.sub_ctx_indices])

        self.scaler_ctx = sklearn.preprocessing.StandardScaler().fit(flattened_ctx_features)
        self.scaler_sub_ctx = sklearn.preprocessing.StandardScaler().fit(flattened_sub_ctx_features)

    def _scale(self, subjects_features):
        """Scale the subject features using constant scaling factor.

        :param subjects_features: The subjects' features to scale [n_sample, n_features].
        :return: The scaled subject's features.
        """
        # TODO(loya, kessi) Validate the normalization.
        res = np.zeros(subjects_features.shape)

        ctx_features = subjects_features[:, self.ctx_indices]
        sub_ctx_features = subjects_features[:, self.sub_ctx_indices]

        flattened_ctx_features = utils.utils.flatten_features_for_scale(ctx_features)
        flattened_sub_ctx_features = utils.utils.flatten_features_for_scale(sub_ctx_features)

        normalize_flattened_ctx_features = self.scaler_ctx.transform(flattened_ctx_features)
        normalize_flattened_sub_ctx_features = self.scaler_sub_ctx.transform(flattened_sub_ctx_features)

        res[:, self.ctx_indices] = normalize_flattened_ctx_features.reshape(ctx_features.shape)
        res[:, self.sub_ctx_indices] = normalize_flattened_sub_ctx_features.reshape(sub_ctx_features.shape)

        return res

    def _get_or_create_semi_dense_connectome_data(self, pca_result, brain_map):
        if not self.semi_dense_connectome_data:
            self.semi_dense_connectome_data = feature_extraction.get_subcortical_parcellation(pca_result, brain_map)
        return self.semi_dense_connectome_data

    def extract(self, subjects, with_scaling=True):
        """Extract the subject features.

        :param subjects: The subjects to extract their features [n_subjects, n_data].
        :param with_scaling:
        :return: The subjects' features.
        """
        # TODO(loya) return to this
        left_right_hemisphere_data = feature_extraction.run_group_ica_separately(
            self.pca_result, self.default_brain_map
        )
        left_right_hemisphere_data = left_right_hemisphere_data.transpose()
        feature_extraction.run_dual_regression(left_right_hemisphere_data, self.default_brain_map, subjects)

        # scipy.io.savemat('dual_reg_result.mat',
        #                  {'dual_reg_result': np.transpose(subjects[0].left_right_hemisphere_data)})
        # subjects[0].left_right_hemisphere_data = np.transpose(
        #     scipy.io.loadmat('dual_reg_result.mat')['dual_reg_result'])
        semi_dense_connectome_data = self._get_or_create_semi_dense_connectome_data(self.pca_result,
                                                                                    self.default_brain_map)
        semi_dense_connectome_data = semi_dense_connectome_data.transpose()
        feature_extraction.get_semi_dense_connectome(semi_dense_connectome_data,
                                                     subjects)
        res = np.array([sub.correlation_coefficient.transpose() for sub in subjects])
        # scipy.io.savemat('feature_ext_result.mat',
        #                  {'feature_ext_result': np.transpose(res)})
        # res = np.transpose(
        #     scipy.io.loadmat('feature_ext_result.mat')['feature_ext_result'])
        if with_scaling:
            return self._scale(res)
        else:
            return res


class Predictor:
    """A class containing all the localizer predictor model data.

        This allow injecting another model instead, as it uses fit(x,y) and predict(x).
    """

    def __init__(self, pca_result):
        """Init the predictor.

        :param pca_result: The pca to extract the spatial filtering from.
                        This is later user to group indexes by their connectivity ICA,
                        and combine them as their group only predictors.
        """
        self.beta = None
        self.spatial_features = feature_extraction.get_spatial_filters(pca_result)

    def _get_beta(self, subject_feature, subject_task):
        """Get the prediction betas from psudo-inverse of ((beta @ [1 subject_features] = subject_task)).

        :param subject_feature: The subject features.
        :param subject_task: The subject task results.
        :return: The subject betas.
        """
        task = subject_task
        subject_feature = sklearn.preprocessing.normalize(subject_feature)
        betas = np.zeros(
            (subject_feature.shape[1] + 1, self.spatial_features.shape[1]))
        for j in range(self.spatial_features.shape[1]):
            ind = self.spatial_features[:, j] > 0
            y = task[ind]
            x = utils.utils.add_ones_column_to_matrix(subject_feature[ind])
            betas[:, j] = np.linalg.pinv(x) @ y
        return betas

    def fit(self, subjects_feature, subjects_task):
        """Fit the model from the data.

        :param subjects_feature: X,
                [n_samples, n_features] Matrix like object containing the subject features.
        :param subjects_task: y,
                [n_samples, n_results] Matrix like object containing the subject task results.
        """
        betas = np.asarray(map(
            lambda feature, task: self._get_beta(
                feature,
                task),
            zip(subjects_feature, subjects_task)
        ), dtype=np.float32)
        self.beta = np.mean(betas, axis=0)

    def predict(self, subjects_features):
        """Predict the task results from the subjects features.

        :param subjects_features: X,
                    [n_subjects, n_features] Matrix like object containing the subjects features.
        :return: y,
                    [n_subjects, n_results] Matrix like object containing the task result prediction.
        """
        if self.beta is None:
            raise BrokenPipeError("Cannot predict before the model was trained!")
        subjects_features_with_ones = map(utils.utils.add_ones_column_to_matrix, subjects_features)

        return map(lambda subject_features: subject_features @ self.beta, subjects_features_with_ones)


class Localizer:
    """A class containing the localizer model data.
    """

    def __init__(self, subjects, subjects_task=None, pca_result=None, predictor=None):
        """Initialize a localizer object

        :param subjects: The subject to train on.
        :param subjects_task: The subject's task result to fit the model on.
        :param pca_result: The pca to use for the features extraction and ICA filtering.
                    If not provided, you must provide subjects to create the PCA from.
        :param feature_extractor: The feature extractor object to use.
                Will be used to extract features from the
        :param predictor: The predictor model to use for prediction.
                    If not provided, a default predictor will be created, and fitted to the subjects,
                    and the subjects' task results.
        """
        if pca_result is None and not subjects:
            raise ValueError("Cannot initialize a localizer if no pca and no subjects were provided, " +
                             "as it cannot generate a new PCA without subjects.")

        if pca_result is None:
            pca_result = iterative_pca.iterative_pca(subjects)

        self._feature_extractor = FeatureExtractor(subjects, pca_result)
        if predictor is None:
            if subjects_task is None:
                raise ValueError(
                    'Cannot initialize a localizer if no predictor was provided, and no subjects and no '
                    'subject\'s_task were provided, as it cannot train a new predictor without subjects and '
                    'subjects\'s task results.')
            predictor = Predictor(pca_result)
            predictor.fit(subjects, subjects_task)
        self._predictor = predictor

    def fit(self, subjects, subjects_task):
        """Fit the current loaded model on the given data.

        :param subjects: The subject to fit on.
        :param subjects_task: The task result of each subject.
        :return:
        """
        subjects_feature = map(self._feature_extractor.extract, subjects)
        self._predictor.fit(subjects_feature, subjects_task)

    def predict(self, subject):
        """Predict the task results from the subjects features.

        :param subject: The subject to predict his task results.
        :return: The task result prediction.
        """
        features = self._feature_extractor.extract(subject)
        return self._predictor.predict(features)

    def save_to_file(self, file_path):
        """Save localizer to file.

        :param file_path: Path to save the object to.
        """
        return pickle.dump(gzip.open(file_path, 'wb'), self)

    @staticmethod
    def load_from_file(file_path):
        """Load a localizer from file.

        :param file_path: File path to load from.
        :return: The localizer object loaded.
        """
        res = pickle.load(gzip.open(file_path, 'rb'))
        if not isinstance(res, Localizer):
            raise TypeError("Content of file is either an old type and deprecated Localizer model, "
                            "a corrupted file or in a wrong file format.")
