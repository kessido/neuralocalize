#!/usr/bin/python

import argparse
import os

import numpy as np
# import sklearn.model_selection
import sklearn

import constants
import utils.cifti_utils
import utils.utils
# TODO(loya) add full description.
from prediction import Localizer, FeatureExtractor

PARSER = argparse.ArgumentParser(
    description=("""
A module for the Localize model.
The model predicts individual differences in brain activity.

This module has two main functionalities:
1. Training a model: Given a task and subject data (rest and task data), trains a model to predict the brain activity in
this task given rest fMRI data.
2. Loading a trained model, and use it to predict task brain activity.

For more information about the model, see Tavor et al. Task-free MRI predicts individual differences
in brain activity during task performance:
http://science.sciencemag.org/content/352/6282/216"""
                 ))

# TODO(loya) extend. Add file types etc.
PARSER.add_argument('--train', action='store_true',
                    help='''Training mode.
                    Given training data (subject rest and task), trains a model and outputs it into the output directory.
                    Requires output_dir, and input_dir with training data files in the following format:
                    ''')  # TODO(loya) decide on a format.
PARSER.add_argument('--predict', action='store_true', help='Predict mode')
PARSER.add_argument('--input_dir', default='./',
                    help='''The path to the input file(s).
                    In training mode, the input files should include subject rest data and task data in the
                    HCP format:
                    The session files are in stored as follows:
                    base_dir | {subject_id} | MNINonLinear | Results | rfMRI_REST{session_num}_{sides} |
                    rfMRI_REST{session_id}_{sides}_Atlas_MSMAll_hp2000_clean.dtseries.nii
                    The task files are stored as follows:
                    base_dir | Tasks
                    In predict mode, only the rest files are required.
                    ''')  # TODO(loya) separate to rest and task, add format
PARSER.add_argument('--output_dir', default='./results',
                    help='''Required for prediction mode. The output folder to use for outputting the predictions.
                    The output will later be saved in the following way:
                    output_dir/
                        {subject}_result.pcl
                    ''')  # Todo(loya) decide on output format for the files.
PARSER.add_argument('--output_filename', default='model.pcl',
                    help='The name to save the output model.')
PARSER.add_argument('--run_with_pca', action='store_true',
                    help='Defaults to False. Set to True if you want to run with PCA, otherwise loads the result.')
PARSER.add_argument('--pca_result', default='../test_resources/GROUP_PCA_rand200_RFMRI.dtseries.nii',
                    help='Optional. Load iterative PCA result from this location.')
PARSER.add_argument('--load_feature_extraction', action='store_true',
                    help='Defaults to False. Set to True if you want to run with feature extraction,'
                         ' otherwise loads the result.')
PARSER.add_argument('--feature_extraction_result', default='../test_resources/feature_ext_result.mat',
                    help='Optional. Load the features from the template path')
PARSER.add_argument('--task_filename', default=constants.DEFAULT_TASK_FILENAME,
                    help='Name of the task files. Stored in {base_dir}/Tasks')
PARSER.add_argument('--task_ordered_subjects_filename', default=constants.DEFAULT_TASK_ORDERED_SUBJ_FILE,
                    help='The path for the file holding a list of the subject ids in the order they appear in the task'
                         ' matrices.')
PARSER.add_argument('--model_file', default=constants.model_filename,
                    help='Required for predict mode. The file containing the trained localizer model is located.')
PARSER.add_argument('--benchmark', action='store_true',
                    help='''For training mode. If presented, a Leave One Out testing method will be used, where the 
                    model will be trained each time excluding one subject, and then testing the model on this subject.
                    The mean prediction accuracy, and std, will then be presented in the console.''')

ARGS = PARSER.parse_args()


def validate_predict_args(args):
    """Validates the prediction arguments are correct.
    :param args:
    :return:
    """
    if not os.path.exists(args.input_dir):
        raise ValueError("Input file doesn't exist.")

    if not os.path.exists(args.model_file):
        raise ValueError("Model file doesn't exist.")


def validate_train_args(args):
    if not os.path.exists(args.input_dir):
        raise ValueError("Input file doesn't exist.")


def load_subjects(args):
    """Load subjects

    :param args:
    :return: [n_subjects, n_data]
    """
    subjects = []
    subj_dir = os.path.join(args.input_dir, 'Subjects')
    subject_folders = os.listdir(subj_dir)
    for subj_folder in subject_folders:
        subj = utils.utils.Subject(name=subj_folder)
        subj.load_from_directory(os.path.join(subj_dir, subj_folder, constants.PATH_TO_SESSIONS))
        subjects.append(subj)
    return subjects


def _get_ordered_subjects_list(path_to_file):
    ret = {}
    with open(path_to_file) as f:
        subjs = f.readlines()
        for i, subj in enumerate(subjs):
            ret[subj.strip()] = i
        return ret


def load_subjects_task(args, subjects):
    """Load subjects' tasks results

    :param args:
    :return: [n_subjects, n_tasks_results]
    """
    tasks_ordered_by_subj = []
    full_path_to_tasks = os.path.join(args.input_dir, 'Tasks', args.task_filename)
    full_path_to_ordered_subjs = os.path.join(args.input_dir, args.task_ordered_subjects_filename)
    subj_index_dict = _get_ordered_subjects_list(full_path_to_ordered_subjs)

    all_subjects_tasks, _ = utils.cifti_utils.load_cifti_brain_data_from_file(full_path_to_tasks)

    for subj in subjects:
        # TODO(loya) this might need transpose:
        tasks_ordered_by_subj.append(all_subjects_tasks[subj_index_dict[subj.name]])
    return tasks_ordered_by_subj


# todo(kess) add optiong to also include PCA
def train_model(subjects, subjects_task, args, pca_result=None, feature_extractor=None):
    """Train a localizer model

    :param subjects: The subjects to train on.
    :param subjects_task: The subjects' task to train on.
    :param args:
    :param pca_result:
    :return: A localizer model
    """
    # todo(kess) this is where you change the model type.
    localizer = Localizer(subjects, pca_result=pca_result,
                          load_feature_extraction=args.load_feature_extraction,
                          feature_extraction_path_template=args.feature_extraction_result, feature_extractor=feature_extractor)
    localizer.fit(subjects, subjects_task)
    return localizer


def get_benchmark(localizer, subjects, subjects_task):
    """Get a benchmark of the localizer on the subjects and subjects_task
    2-norm mean.

    :param localizer:
    :param subjects: The subjects to test on.
    :param subjects_task: The subjects' task to test on.
    :return: The benchmark.
    """
    predictions = localizer.predict(subjects)
    res = []
    for subject_task, prediction in zip(subjects_task, predictions):
        dif = np.abs(np.array(subject_task) - np.array(prediction))
        res.append(np.linalg.norm(dif))
    return np.mean(np.array(res))


def benchmark(subjects, subjects_task, args, pca_result=None):
    benchmark_list = []
    feature_extractor = FeatureExtractor(subjects, pca_result)
    subjects_task = np.array(subjects_task)
    subjects = np.array(subjects)
    for train_indices, test_indices in sklearn.model_selection.LeaveOneOut().split(subjects):
        print("Run on ", train_indices, test_indices)
        localizer = train_model(subjects[train_indices],
                                subjects_task[train_indices],
                                args, pca_result, feature_extractor)
        benchmark_list.append(
            get_benchmark(localizer,
                          subjects[test_indices],
                          subjects_task[test_indices]))

    return np.mean(benchmark_list), np.std(benchmark_list), benchmark_list


def main():
    if ARGS.train and not ARGS.predict:
        validate_train_args(ARGS)
        print("Loading subjects and tasks.")
        subjects = load_subjects(ARGS)
        subjects_task = load_subjects_task(ARGS, subjects)
        pca_result = None
        if not ARGS.run_with_pca:
            pca_result_path = os.path.join(ARGS.input_dir, ARGS.pca_result)
            pca_result, _ = utils.cifti_utils.load_cifti_brain_data_from_file(pca_result_path)

        if ARGS.benchmark:
            if len(subjects) <= 1:
                raise ValueError("Not enough subjects to preform leave-one-out.")
            print("Benchmark Starting...")
            mean, std, raw = benchmark(subjects, subjects_task, ARGS, pca_result=pca_result)
            print("Benchmark Results:")
            print("Mean:", mean)
            print("STD:", std)
            print("Raw data:", raw)

        output_path = os.path.join(ARGS.output_dir, ARGS.output_filename)
        print("Training Model.")
        train_model(subjects, subjects_task, ARGS, pca_result=pca_result).save_to_file(output_path)
        print("Finished.")

    elif ARGS.predict and not ARGS.train:
        validate_predict_args(ARGS)
        print("Loading subjects and tasks.")
        subjects = load_subjects(ARGS)
        print("Loading model.")
        localizer = Localizer.load_from_file(ARGS.model_file)
        print("Running predictions.")
        predictions = localizer.predict(subjects, load_feature_extraction=ARGS.load_feature_extraction,
                                        feature_extraction_path=ARGS.feature_extraction_result)
        utils.utils.create_dir(ARGS.output_dir)
        print("Saving Results.")
        utils.cifti_utils.save_cifti(predictions, os.path.join(ARGS.output_dir, 'result.dtseries.nii'))
        print("Finished")
    else:
        PARSER.print_help()
        raise ValueError("Either --train or --predict must be provided, and not both.")


if __name__ == '__main__':
    main()
