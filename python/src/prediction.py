"""This code simulates the prediction code of the of the connectivity model.
"""

import gzip
import pickle
from constants import dtype
import numpy as np
import sklearn.preprocessing

import feature_extraction
import iterative_pca
import utils.cifti_utils
import utils.utils


class Localizer:
    """A class containing the localizer model data.
    """

    class FeatureExtractor:
        """A class warping the scaling and feature extraction methods.
        """

        def __init__(self, subjects, pca):
            """ Init the Feature Extractor from subjects and pca.
            Create a scaling factor of the cortical and sub cortical parts.

            :param subjects: The subjects
            :param pca: the PCA  to use.
            """
            self.ctx_indices, self.sub_ctx_indices = \
                utils.cifti_utils.get_cortex_and_sub_cortex_indices()
            features = feature_extraction.extract_features(subjects, pca)
            features = np.asarray(features)
            self.pca = pca
            # TODO(kess) check if this doesn't just return the same result as scaling by the whole thing.
            self.scaler_ctx = sklearn.preprocessing. \
                StandardScaler().fit(features[:, self.ctx_indices])
            self.scaler_sub_ctx = sklearn.preprocessing. \
                StandardScaler().fit(features[:, self.sub_ctx_indices])

        def _scale(self, subjects_features):
            """Scale the subject features using constant scaling factor.

            :param subjects_features: The subjects' features to scale [n_sample, n_features].
            :return: The scaled subject's features.
            """
            res = np.zeros(subjects_features.shape)
            res[:, self.ctx_indices] = self.scaler_ctx.transform(subjects_features[:, self.ctx_indices])
            res[:, self.sub_ctx_indices] = self.scaler_sub_ctx.transform(subjects_features[:, self.sub_ctx_indices])
            return res

        def extract(self, subjects):
            """Extract the subject features.

            :param subjects: The subjects to extract their features [n_subjects, n_data].
            :return: The subjects' features.
            """
            res = np.asarray(feature_extraction.extract_features(subjects, self.pca))
            return self._scale(res)

    class Predictor:
        """A class containing all the localizer predictor model data.

            This allow injecting another model instead, as it uses fit(x,y) and predict(x).
        """

        def __init__(self, pca):
            """Init the predictor.

            :param pca: The pca to extract the spatial filtering from.
                            This is later user to group indexes by their connectivity ICA,
                            and combine them as their group only predictors.
            """
            self.beta = None
            self.spatial_features = feature_extraction.get_spatial_filters(pca)

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
            ), dtype=dtype)
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

    def __init__(self, subjects, subjects_task=None, pca=None, predictor=None):
        """Initialize a localizer object

        :param subjects: The subject to train on.
        :param subjects_task: The subject's task result to fit the model on.
        :param pca: The pca to use for the features extraction and ICA filtering.
                    If not provided, you must provide subjects to create the PCA from.
        :param feature_extractor: The feature extractor object to use.
                Will be used to extract features from the
        :param predictor: The predictor model to use for prediction.
                    If not provided, a default predictor will be created, and fitted to the subjects,
                    and the subjects' task results.
        """
        if pca is None:
            raise ValueError("Cannot initialize a localizer if no pca and no subjects were provided, " +
                             "as it cannot generate a new PCA without subjects.")
        pca = iterative_pca.iterative_pca(subjects)

        self._feature_extractor = Localizer.FeatureExtractor(subjects, pca)
        if predictor is None:
            if subjects_task is None:
                raise ValueError(
                    'Cannot initialize a localizer if no predictor was provided, and no subjects and no '
                    'subject\'s_task were provided, as it cannot train a new predictor without subjects and '
                    'subjects\'s task results.')
            predictor = Localizer.Predictor(pca)
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

#
# def run_all_predictions(subject_data_dir, result_dir, subjects, task_path, path_to_something):
#     """
#     docstring here
#         :param subject_data_dir:
#         :param result_dir:
#         :param subjects:
#         :param task_path:
#         :param path_to_something:
#     """
#     datadir = subject_data_dir
#     outdir = result_dir
#     create_dir(outdir)
#     # addpath('./extras', './extras/CIFTIMatlabReaderWriter')
#     # % % % % % % % % % % % % % % % % % % % % % % % % % % %
#     # subjects = textread('./extras/subjects.txt', '%s')
#     features_dir = os.path.join(result_dir, 'Features')
#     # task_path = '/path/to/task/contrasts'
#     # the task contrasts should be CIFTI files as described below
#     # task_path = '/vols/Scratch/saad/MVPA_Functional_Localisation/Tasks'
#     # path_to_something = './extras/CIFTIMatlabReaderWriter/example.dtseries.nii'
#     ctx_indexes, sub_ctx_indexes = utils.cifti_utils.get_cortex_and_sub_cortex_indices()
#
#     # Load all Features from all subjects(needs RAM)
#     print('Load All Features')
#     all_features = map(
#         lambda x: load_subject_features(x, features_dir), subjects)
#     all_features = np.asarray(all_features)
#
#     # normalise features
#     all_features = np.transpose(all_features, [1, 2, 0])
#     all_features[ctx_indexes] = normalize(all_features[ctx_indexes])
#     all_features[sub_ctx_indexes] = normalize(all_features[sub_ctx_indexes])
#     all_features = np.transpose(all_features, [2, 0, 1])
#
#     # % % load all tasks
#     # % Task contracts are CIFTI files
#     # % One file per contrast, with all subjects concatenated in each file
#     nsubjects = len(subjects)
#     ncontrasts = 86
#     # % these contain both pos and neg contrast maps. There are 47 independent contrasts in there
#     all_tasks = np.zeros((ncontrasts, nsubjects, 91282))
#     for i in range(ncontrasts):
#         print(f"contrasts {i}")
#         task_path_all_tasks = os.path.join(
#             task_path, f'AllSubjects_{i:03}.dtseries.nii')
#         all_tasks[i] = load_nii_brain_image_from_file(task_path_all_tasks)
#
#     # % % Load spatial filters
#     # % then threshold and do a winner-take-all
#     print('Load Filters')
#     filter_path = os.path.join(outdir, const.filter_filename)
#     filters = load_nii_brain_image_from_file(filter_path)
#     # m, wta = max(filters, [], 2)
#     m = np.max(filter, axis=1)
#     wta = np.argmax(filter, axis=1)
#     wta = wta * (m > 2.1)
#     S = np.zeros(filters.shape)
#     for i in range(filters.shape[1]):
#         S[:, i] = np.asarray(wta == i)
#
#     # % % Run training and LOO predictions
#     print('Start')
#     betas_dir = os.path.join(outdir, 'Betas')
#     predictions_dir = os.path.join(outdir, 'Predictions')
#     create_dir(betas_dir)
#     create_dir(predictions_dir)
#
#     for contrastNum in range(ncontrasts):
#         print(f'contrast {contrastNum:03}')
#         # % start learning
#         print('--> start Learning')
#         feat_idx = range(all_features.shape[2])
#         for i, subject in enumerate(subjects):
#             print(f'\t{i:03}:{subject}')
#             subject_task = all_tasks[contrastNum, i, :]
#             subject_task_ones = np.ones((subject_task.shape[0], 1))
#             subject_normalize_features = normalize(all_features[i, feat_idx])
#             subject_task_features = np.concatenate(
#                 subject_task_ones, subject_normalize_features)
#             # % do the GLM
#             betas = np.zeros(subject_task_features.shape[1], S.shape[1])
#             for j in range(S.shape[1]):
#                 ind = S[j] > 0
#                 y = subject_task[ind]
#                 current_features = subject_task_features[ind, :]
#                 current_features_without_first = subject_task_features[:, 1:]
#                 current_features_without_first_demean = current_features_without_first - \
#                                                         np.mean(current_features_without_first, keepdims=True)
#
#                 M = np.concatenate(
#                     current_features[0], current_features_without_first_demean)
#                 betas[:, j] = np.linalg.pinv(M) * y
#             beta_save_path = os.path.join(betas_dir, f'contrast_{contrastNum:03}_{subject}_betas.mat')
#             with open(beta_save_path, 'wb') as pfile:
#                 pickle.dump(betas, pfile, protocol=pickle.HIGHEST_PROTOCOL)
#             print('\t\tsaved beta')
#
#         # % leave-one-out betas
#         print('--> LOO!')
#         for i, subject in enumerate(subjects):
#             print(f'\t{i:03}:{subject}')
#             loo_betas = np.zeros((all_features.shape[1], S.shape[1], len(subjects) - 1))
#             cnt = 1
#             for j, subjectj in enumerate(subjects):
#                 if i == j:
#                     continue
#                 beta_load_path = os.path.join(betas_dir, f'contrast_{contrastNum:03}_{subjectj}_betas.mat')
#                 with open(beta_load_path, 'rb') as pfile:
#                     betas = pickle.load(pfile)
#                 loo_betas[:, :, cnt] = betas
#                 cnt = cnt + 1
#             loo_beta_save_path = os.path.join(betas_dir, f'contrast_{contrastNum:03}_{subject}_loo_betas.mat')
#             with open(loo_beta_save_path, 'wb') as pfile:
#                 pickle.dump(loo_betas, pfile, protocol=pickle.HIGHEST_PROTOCOL)
#             print('\t\tsaved loo_beta')
#
#         # % predict
#         print('--> Predict!')
#         # todo(kess) - what is 98?
#         X = np.zeros((91282, 98))
#         for i, subject in enumerate(subjects):
#             print(f'\t{i:03}:{subject}')
#             subject_task = all_tasks[:, i]
#             subject_task_ones = np.ones((subject_task.shape[0], 1))
#             subject_normalize_features = normalize(all_features[i, feat_idx])
#             subject_task_features = np.concatenate(
#                 subject_task_ones, subject_normalize_features)
#             loo_beta_load_path = os.path.join(betas_dir, f'contrast_{contrastNum:03}_{subject}_loo_betas.mat')
#             with open(loo_beta_load_path, 'rb') as pfile:
#                 loo_betas = pickle.load(pfile)
#             pred = 0 * subject_task
#             for j in range(S.shape[1]):
#                 ind = S[:, j] > 0
#                 current_features = subject_task_features[ind, :]
#                 current_features_without_first = subject_task_features[:, 1:]
#                 current_features_without_first_demean = current_features_without_first - \
#                                                         np.mean(current_features_without_first, keepdims=True)
#
#                 M = np.concatenate(
#                     current_features[ind, 0], current_features_without_first_demean[ind])
#                 pred[ind] = M * np.mean(loo_betas[:, j, :], 3)
#             X[:, i] = pred
#         pred_loo_save_path = os.path.join(predictions_dir, f'contrast_{contrastNum:03}_pred_loo.dtseries.nii')
#         cifti.save_image_to_file(X, pred_loo_save_path)


# def load_subject_features(subject, features_dir):
#     subject_dir = os.path.join(features_dir, subject)
#
#     # functional connectivity
#     fc_features_files_path = os.path.join(
#         subject_dir, const.RFMRI_nosmoothing_filename)
#     fc_features, _ = utils.cifti_utils.load_nii_brain_data_from_file(fc_features_files_path)
#     fc_features = np.asarray(fc_features)
#
#     return fc_features
