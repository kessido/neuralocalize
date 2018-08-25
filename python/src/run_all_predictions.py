"""This code simulates the predictoin code of the of the connectivity model.
"""

from ..utils.folders import *
from ..utils.cifti_utils import *
from ..utils.list import *
import numpy as np
import os
from sklearn.preprocessing import normalize
from .. import constants as const
import itertools


# todo(kess) make test for this part.

def load_subject_features(subject, features_dir):
    subject_dir = os.path.join(features_dir, subject)

    # functional connectivity
    fc_features_path = os.path.join(subject_dir, const.RFMRI_nosmoothing_filename)
    fc_features = load_nii_brain_image_from_file(fc_features_path)
    fc_features = np.asarray(fc_features)

    # structural features
    str_features_path = os.path.join(features_dir, subject, const.struct_filename)
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
    ctx_indexes = itertools.chain(brain_maps[0].data_indices, brain_maps[1].data_indices)
    sub_ctx_indexes = remove_list_from_list(range(91282), ctx_indexes)

    # Load all Features from all subjects(needs RAM)
    print('Load All Features')
    all_features = map(lambda x: load_subject_features(x, features_dir), subjects)
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
        task_path_all_tasks = os.path.join(task_path, f'AllSubjects_{i:03}.dtseries.nii')
        all_tasks[i] = load_nii_brain_image_from_file(path_to_something)

    # % % Load spatial filters
    # % then threshold and do a winner-take-all
    print('Load Filters')
    filter_path = os.path.join(outdir, const.filter_filename)
    filters = load_nii_brain_image_from_file(filter_path)
    [m, wta] = max(filters, [], 2)
    wta = wta .* (m > 2.1)
    S = zeros(size(filters.cdata))
    for i = 1:
        size(filters.cdata, 2)
        S(: , i) = double(wta == i)
    end

        # % % Run training and LOO predictions
        # disp('Start')
        # unix(['mkdir -p ' outdir '/Betas'])
        # unix(['mkdir -p ' outdir '/Predictions'])

        # for contrastNum = 1:
        #     86
        #     disp(['contrast ' zeropad(contrastNum, 3)])

        #     % start learning
        #     disp('--> start Learning')
        #     featIdx = 1: size(all_features, 3)

        #     for i = 1:
        #         length(subjects)
        #         subj = subjects{i}
        #         disp(['      ' subj])
        #         task = AllTasks(: , i, contrastNum)
        #         Features = [ones(size(task, 1), 1) normalise(squeeze(all_features(:, i, featIdx)))]
        #         % do the GLM
        #         betas = zeros(size(Features, 2), size(S, 2))
        #         for j = 1:
        #             size(S, 2)
        #             ind = S(: , j) > 0
        #             y = task(ind)
        #             M = [Features(ind, 1) demean(Features(ind, 2:end))]
        #             betas(:, j) = pinv(M)*y
        #         end
        #         save([outdir '/Betas/contrast_' zeropad(contrastNum, 3) '_' subj '_betas.mat'], 'betas')
        #     end

        #     % leave-one-out betas
        #     disp('--> LOO')
        #     for i = 1:
        #         length(subjects)
        #         subj = subjects{i}
        #         disp(['      ' subj])
        #         loo_betas = zeros(size(Features, 2), size(S, 2), length(subjects)-1)
        #         cnt = 1
        #         for j = setdiff(1: length(subjects), i)
        #             load([outdir '/Betas/contrast_' zeropad(contrastNum, 3) '_' subjects{j} '_betas.mat'], 'betas')
        #             loo_betas(:, : , cnt) = betas
        #             cnt = cnt+1
        #         end
        #         save([outdir '/Betas/contrast_' zeropad(contrastNum, 3) '_' subjects{i} '_loo_betas.mat'], 'loo_betas')
        #     end

        #     disp('--> Predict')
        #     % predict
        #     X = zeros(91282, 98)
        #     for i = 1:
        #         length(subjects)
        #         subj = subjects{i}
        #         disp(['      ' subj])
        #         task = AllTasks(: , i)
        #         Features = [ones(size(task, 1), 1) normalise(squeeze(all_features(:, i, featIdx)))]
        #         load([outdir '/Betas/contrast_' zeropad(contrastNum, 3) '_' subj '_loo_betas.mat'], 'loo_betas')

        #         pred = 0*task
        #         for j = 1:
        #             size(S, 2)
        #             ind = S(: , j) > 0
        #             M = [Features(ind, 1) demean(Features(ind, 2:end))]
        #             pred(ind) = M*mean(loo_betas(: , j, : ), 3)
        #         end
        #         X(:, i) = pred

        #     end
        #     cifti.cdata = X
        #     ciftisave(cifti, [outdir '/Predictions/contrast_' zeropad(contrastNum, 3) '_pred_loo.dtseries.nii'])


        # end
