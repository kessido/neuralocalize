"""This code simulates the predictoin code of the of the connectivity model.
"""

import pickle
from utils.folders import *
from utils.cifti_utils import *
from utils.list import *
import numpy as np
import os
from sklearn.preprocessing import normalize
import constants as const
import itertools


# todo(kess) make test for this part.
def load_subject_features(subject, features_dir):
    subject_dir = os.path.join(features_dir, subject)

    # functional connectivity
    fc_features_path = os.path.join(
        subject_dir, const.RFMRI_nosmoothing_filename)
    fc_features = load_nii_brain_image_from_file(fc_features_path)
    fc_features = np.asarray(fc_features)

    # structural features
    str_features_path = os.path.join(
        features_dir, subject, const.struct_filename)
    str_features = load_nii_brain_image_from_file(str_features_path)
    str_features = np.asarray(str_features, dtype=np.float32)

    return np.concatenate((fc_features, str_features))


def run_all_predictions(subject_data_dir, result_dir, subjects, task_path, path_to_something):
    """
    docstring here
        :param subject_data_dir:
        :param result_dir:
        :param subjects:
        :param task_path:
        :param path_to_something:
    """
    datadir = subject_data_dir
    outdir = result_dir
    create_dir(outdir)
    # addpath('./extras', './extras/CIFTIMatlabReaderWriter')
    # % % % % % % % % % % % % % % % % % % % % % % % % % % %
    # subjects = textread('./extras/subjects.txt', '%s')
    features_dir = os.path.join(result_dir, 'Features')
    # task_path = '/path/to/task/contrasts'
    # the task contrasts should be CIFTI files as described below
    # task_path = '/vols/Scratch/saad/MVPA_Functional_Localisation/Tasks'
    # path_to_something = './extras/CIFTIMatlabReaderWriter/example.dtseries.nii'
    cifti, brain_maps = load_nii_brain_data_from_file(path_to_something)
    ctx_indexes = itertools.chain(
        brain_maps[0].data_indices, brain_maps[1].data_indices)
    sub_ctx_indexes = remove_list_from_list(range(91282), ctx_indexes)

    # Load all Features from all subjects(needs RAM)
    print('Load All Features')
    all_features = map(
        lambda x: load_subject_features(x, features_dir), subjects)
    all_features = np.asarray(all_features)

    # normalise features
    all_features = np.transpose(all_features, [1, 2, 0])
    all_features[ctx_indexes] = normalize(all_features[ctx_indexes])
    all_features[sub_ctx_indexes] = normalize(all_features[sub_ctx_indexes])
    all_features = np.transpose(all_features, [2, 0, 1])

    # % % load all tasks
    # % Task contracts are CIFTI files
    # % One file per contrast, with all subjects concatenated in each file
    nsubjects = len(subjects)
    ncontrasts = 86
    # % these contain both pos and neg contrast maps. There are 47 independent contrasts in there
    all_tasks = np.zeros((ncontrasts, nsubjects, 91282))
    for i in range(ncontrasts):
        print(f"contrasts {i}")
        task_path_all_tasks = os.path.join(
            task_path, f'AllSubjects_{i:03}.dtseries.nii')
        all_tasks[i] = load_nii_brain_image_from_file(task_path_all_tasks)

    # % % Load spatial filters
    # % then threshold and do a winner-take-all
    print('Load Filters')
    filter_path = os.path.join(outdir, const.filter_filename)
    filters = load_nii_brain_image_from_file(filter_path)
    # m, wta = max(filters, [], 2)
    m = np.max(filter, axis=1)
    wta = np.argmax(filter, axis=1)
    wta = wta * (m > 2.1)
    S = np.zeros(filters.shape)
    for i in range(filters.shape[1]):
        S[:, i] = np.asarray(wta == i)

    # % % Run training and LOO predictions
    print('Start')
    betas_dir = os.path.join(outdir, 'Betas')
    predictions_dir = os.path.join(outdir, 'Predictions')
    create_dir(betas_dir)
    create_dir(predictions_dir)

    for contrastNum in range(ncontrasts):
        print(f'contrast {contrastNum:03}')
        # % start learning
        print('--> start Learning')
        feat_idx = range(all_features.shape[2])
        for i, subject in enumerate(subjects):
            print(f'\t{i:03}:{subject}')
            subject_task = all_tasks[contrastNum, i, :]
            subject_task_ones = np.ones((subject_task.shape[0], 1))
            subject_normalize_features = normalize(all_features[i, feat_idx])
            subject_task_features = np.concatenate(
                subject_task_ones, subject_normalize_features)
            # % do the GLM
            betas = np.zeros(subject_task_features.shape[1], S.shape[1])
            for j in range(S.shape[1]):
                ind = S[j] > 0
                y = subject_task[ind]
                current_features = subject_task_features[ind, :]
                current_features_without_first = subject_task_features[:, 1:]
                current_features_without_first_demean = current_features_without_first - \
                                                        np.mean(current_features_without_first, keepdims=True)

                M = np.concatenate(
                    current_features[0], current_features_without_first_demean)
                betas[:, j] = np.linalg.pinv(M) * y
            beta_save_path = os.path.join(betas_dir, f'contrast_{contrastNum:03}_{subject}_betas.mat')
            with open(beta_save_path, 'wb') as pfile:
                pickle.dump(betas, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('\t\tsaved beta')

        # % leave-one-out betas
        print('--> LOO!')
        for i, subject in enumerate(subjects):
            print(f'\t{i:03}:{subject}')
            loo_betas = np.zeros((all_features.shape[1], S.shape[1], len(subjects) - 1))
            cnt = 1
            for j, subjectj in enumerate(subjects):
                if i == j:
                    continue
                beta_load_path = os.path.join(betas_dir, f'contrast_{contrastNum:03}_{subjectj}_betas.mat')
                with open(beta_load_path, 'rb') as pfile:
                    betas = pickle.load(pfile)
                loo_betas[:, :, cnt] = betas
                cnt = cnt + 1
            loo_beta_save_path = os.path.join(betas_dir, f'contrast_{contrastNum:03}_{subject}_loo_betas.mat')
            with open(loo_beta_save_path, 'wb') as pfile:
                pickle.dump(loo_betas, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('\t\tsaved loo_beta')

        # % predict
        print('--> Predict!')
        # todo(kess) - what is 98?
        X = np.zeros((91282, 98))
        for i, subject in enumerate(subjects):
            print(f'\t{i:03}:{subject}')
            subject_task = all_tasks[:, i]
            subject_task_ones = np.ones((subject_task.shape[0], 1))
            subject_normalize_features = normalize(all_features[i, feat_idx])
            subject_task_features = np.concatenate(
                subject_task_ones, subject_normalize_features)
            loo_beta_load_path = os.path.join(betas_dir, f'contrast_{contrastNum:03}_{subject}_loo_betas.mat')
            with open(loo_beta_load_path, 'rb') as pfile:
                loo_betas = pickle.load(pfile)
            pred = 0 * subject_task
            for j in range(S.shape[1]):
                ind = S[:, j] > 0
                current_features = subject_task_features[ind, :]
                current_features_without_first = subject_task_features[:, 1:]
                current_features_without_first_demean = current_features_without_first - \
                                                        np.mean(current_features_without_first, keepdims=True)

                M = np.concatenate(
                    current_features[ind, 0], current_features_without_first_demean[ind])
                pred[ind] = M * np.mean(loo_betas[:, j, :], 3)
            X[:, i] = pred
        pred_loo_save_path = os.path.join(predictions_dir, f'contrast_{contrastNum:03}_pred_loo.dtseries.nii')
        cifti.save_image_to_file(X, pred_loo_save_path)
