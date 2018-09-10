"""This code simulates the prediction code of the of the connectivity model.
"""

import gzip
import pickle
import uuid

import numpy as np
import scipy.io
import sklearn.preprocessing

import feature_extraction
import iterative_pca
import utils.cifti_utils
import utils.utils


class FeatureExtractor:
    """A class warping the scaling and feature extraction methods.
    """

    # todo add option to create the pca results.
    def __init__(self, subjects, pca_result=None, sample_file_path='../resources/example.dtseries.nii',
                 load_feature_extraction=False,
                 feature_extraction_path=''):
        """ Init the Feature Extractor from subjects and pca.
        Create a scaling factor of the cortical and sub cortical parts.

        :param subjects: The subjects
        :param pca_result: the PCA  to use.
        """
        self.semi_dense_connectome_data = None
        self.left_right_hemisphere_data = None
        self.ctx_normalizer = utils.utils.Normalizer()
        self.sub_ctx_normalizer = utils.utils.Normalizer()
        self.uuid = uuid.uuid4()

        if pca_result is None:
            raise NotImplementedError("Not yet supported")

        self.pca_result = pca_result

        _, self.default_brain_map = utils.cifti_utils.load_cifti_brain_data_from_file(sample_file_path)

        self.ctx_indices, self.sub_ctx_indices = utils.cifti_utils.get_cortex_and_sub_cortex_indices(sample_file_path)
        # [subjects x features (tasks x brain)]
        print("Extracting features.")
        features = self.extract(subjects, False, load_feature_extraction,
                                feature_extraction_path)
        features = np.array(features)
        # TODO(kess) check if this doesn't just return the same result as scaling by the whole thing.
        features = self._scale_replace(features)

        self._add_features_to_subjects(subjects, features)

    def _add_features_to_subjects(self, subjects, features):
        for subject, feature in zip(subjects, features):
            subject.features_extractor_uuid = self.uuid
            subject.features = feature

    def _scale_replace(self, features):

        print("features before transpose:", features.shape)
        features = np.transpose(features, [1, 0, 2])
        print("features after:", features.shape)
        print("SHAPE:", features[self.ctx_indices, :, :].shape)
        ctx_features = features[self.ctx_indices, :, :]
        sub_ctx_features = features[self.sub_ctx_indices, :, :]
        if not self.ctx_normalizer.is_fit:
            normalized_ctx_features = self.ctx_normalizer.fit(ctx_features)
        else:
            normalized_ctx_features = self.ctx_normalizer.normalize(ctx_features)
        if not self.sub_ctx_normalizer.is_fit:
            normalized_sub_ctx_features = self.sub_ctx_normalizer.fit(sub_ctx_features)
        else:
            normalized_sub_ctx_features = self.sub_ctx_normalizer.normalize(sub_ctx_features)
        features[self.ctx_indices, :, :] = normalized_ctx_features
        features[self.sub_ctx_indices, :, :] = normalized_sub_ctx_features
        features = np.transpose(features, [1, 0, 2]) # TODO(loya) experiment
        return features

    # TODO(loya) delete and validate nothing breaks.
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

    def _get_or_create_semi_dense_connectome_data(self):
        if self.semi_dense_connectome_data is None:
            self.semi_dense_connectome_data = feature_extraction.get_subcortical_parcellation(
                self.pca_result, self.default_brain_map)
        return self.semi_dense_connectome_data

    def _get_or_create_left_right_hemisphere_data(self):
        if self.left_right_hemisphere_data is None:
            self.left_right_hemisphere_data = feature_extraction.run_group_ica_separately(
                self.pca_result, self.default_brain_map
            ).transpose()
        return self.left_right_hemisphere_data

    def extract(self, subjects, with_scaling=True, load_feature_extraction=False,
                feature_extraction_path=''):
        """Extract the subject features.

        :param subjects: The subjects to extract their features [n_subjects, n_data].
        :param load_feature_extraction:
        :param feature_extraction_path:
        :param with_scaling:
        :return: The subjects' features.
        """
        if load_feature_extraction:
            features, _ = utils.cifti_utils.load_cifti_brain_data_from_file(
                r'..\test_resources\100307_RFMRI_nosmoothing.dtseries.nii'
            )
            res = [np.transpose(features)]
            # return np.transpose(
            #     scipy.io.loadmat(feature_extraction_path)['feature_ext_result'])

        else:
            res = [None for _ in range(len(subjects))]

            subjects_not_computed_indexes = []
            needed_subjects = []

            for i, subject in enumerate(subjects):
                if subject.features_extractor_uuid == self.uuid:
                    res[i] = subject.features
                else:
                    subjects_not_computed_indexes.append(i)
                    needed_subjects.append(subject)

            subjects = needed_subjects

            if len(subjects) > 0:
                feature_extraction.run_dual_regression(self._get_or_create_left_right_hemisphere_data(),
                                                       self.default_brain_map, subjects)

                semi_dense_connectome_data = self._get_or_create_semi_dense_connectome_data().transpose()
                feature_extraction.get_semi_dense_connectome(semi_dense_connectome_data, subjects)
                feature_extraction_res = [sub.correlation_coefficient.transpose() for sub in subjects]
                for i, subject_result in zip(subjects_not_computed_indexes, feature_extraction_res):
                    res[i] = subject_result
        res = np.array(res)
        if with_scaling:
            res = self._scale_replace(res)
        self._add_features_to_subjects(subjects, res)
        return res


class Predictor:
    """A class containing all the localizer predictor model data.

        This allow injecting another model instead, as it uses fit(x,y) and predict(x).
    """

    def __init__(self, pca_result, brain_maps, load_ica_result):
        """Init the predictor.

        :param pca_result: The pca to extract the spatial filtering from.
                        This is later user to group indexes by their connectivity ICA,
                        and combine them as their group only predictors.
        """
        self.betas = None
        self.spatial_filters = feature_extraction.get_spatial_filters(pca_result, brain_maps, load_ica_result)

    def _get_beta(self, subject_features, subject_task):
        """Get the prediction betas from psudo-inverse of ((beta @ [1 subject_features] = subject_task)).

        :param subject_features: The subject features.
        :param subject_task: The subject task results.
        :return: The subject betas.
        """
        task = subject_task
        # TODO(loya) do we get this before or after the transposition?
        normalizer = utils.utils.Normalizer()
        subject_features = normalizer.fit(subject_features)
        betas = np.zeros(
            (subject_features.shape[1] + 1, self.spatial_filters.shape[1]))
        for j in range(self.spatial_filters.shape[1]):
            ind = self.spatial_filters[:, j] > 0
            if np.any(ind):
                y = task[ind]
                demeaned_features = sklearn.preprocessing.scale(subject_features[ind], with_std=False)
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
        betas = []
        for subject_feature, task in zip(subjects_feature, subjects_task):
            betas.append(self._get_beta(subject_feature, task))
        betas = np.array(betas, dtype=np.float32)
        self.betas = betas

    def _predict(self, subject_features):
        res = np.zeros(self.spatial_filters.shape[0])
        betas_after_transpose = self.betas.transpose([1, 2, 0])  # TODO(loya) added the transposition to the scaling
        for j in range(self.spatial_filters.shape[1]):
            ind = self.spatial_filters[:, j] > 0
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
        if self.betas is None:
            raise BrokenPipeError("Cannot predict before the model was trained!")
        return np.array([self._predict(subject_features) for subject_features in subjects_features])


class Localizer:
    """A class containing the localizer model data.
    """

    def __init__(self, subjects, subjects_task=None, pca_result=None, predictor=None,
                 load_feature_extraction=False,
                 feature_extraction_path='', feature_extractor=None, load_ica_result=False):
        """Initialize a localizer object

        :param subjects: The subject to train on.
        :param subjects_task: The subjects' task result to fit the model on.
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

        if feature_extractor is None:
            feature_extractor = FeatureExtractor(subjects, pca_result,
                                                 load_feature_extraction=load_feature_extraction,
                                                 feature_extraction_path=feature_extraction_path)

        self._feature_extractor = feature_extractor

        if predictor is None:
            if subjects_task is None:
                raise ValueError(
                    'Cannot initialize a localizer if no predictor was provided, and no subjects and no '
                    'subject\'s_task were provided, as it cannot train a new predictor without subjects and '
                    'subjects\'s task results.')
            predictor = Predictor(pca_result, self._feature_extractor.default_brain_map, load_ica_result=load_ica_result)
            subjects_features = [subject.features for subject in subjects]
            print("Fitting predictor")
            predictor.fit(subjects_features, subjects_task)
        self._predictor = predictor

    def fit(self, subjects, subjects_task, load_feature_extraction=False,
            feature_extraction_path=''):
        """Fit the current loaded model on the given data.

        :param subjects: The subject to fit on.
        :param subjects_task: The task result of each subject.
        :return:
        """
        subjects_feature = self._feature_extractor.extract(subjects, load_feature_extraction=load_feature_extraction,
                                                           feature_extraction_path=feature_extraction_path)
        self._predictor.fit(subjects_feature, subjects_task)

    def predict(self, subject, load_feature_extraction=False,
                feature_extraction_path=''):
        """Predict the task results from the subjects features.

        :param subject: The subject to predict his task results.
        :return: The task result prediction.
        """
        print("load_feature_extraction?", load_feature_extraction)
        features = self._feature_extractor.extract(subject, load_feature_extraction=load_feature_extraction,
                                                   feature_extraction_path=feature_extraction_path)
        res = self._predictor.predict(features)
        return res

    def save_to_file(self, file_path):
        """Save localizer to file.

        :param file_path: Path to save the object to.
        """
        return pickle.dump(self, gzip.open(file_path, 'wb'))

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
        return res
