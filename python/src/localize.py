#!/usr/bin/python

import argparse
import os
import pickle

import numpy as np
import sklearn.model_selection

import constants
from constants import dtype
import utils.utils
# TODO(loya) add full description.
from prediction import Localizer

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
PARSER.add_argument('--predict', action='store_true', )
PARSER.add_argument('--input_dir', default='./',
                    help='''The path to the input file(s).
                    In training mode, the input files should include subject rest data and task data in the
                    following format:
                    <TODO(loya) format>.
                    In predict mode, only the rest files are required.
                    ''')  # TODO(loya) separate to rest and task, add format
PARSER.add_argument('--output_dir', default=f'./results',
                    help='''Required for prediction mode. The output folder to use for outputting the predictions.
                    The output will later be saved in the following way:
                    output_dir/
                        {subject}_result.pcl
                    ''')  # Todo(loya) decide on output format for the files.
PARSER.add_argument('--output_file', default=f'./{constants.model_filename}',
                    help='For training mode. The file where the model will be written to. It is ')
PARSER.add_argument('--model_file', default=f'./{constants.model_filename}',
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

    if os.path.exists(args.model_file):
        raise ValueError("Model file doesn't exist.")

    if os.path.exists(args.output_dir):
        raise ValueError("Output directory already exist. Please delete it first, or change the output directory.")


def validate_train_args(args):
    if not os.path.exists(args.input_dir):
        raise ValueError("Input file doesn't exist.")

    if os.path.exists(args.output_file):
        raise ValueError("Output file already exist. Please delete it first, or change the output file.")


def load_subjects(args):
    """Load subjects

    :param args:
    :return: [n_subjects, n_data]
    """
    pass  # todo(loya) load subjects by folder structures.


def load_subjects_task(args):
    """Load subjects' tasks results

    :param args:
    :return: [n_subjects, n_tasks_results]
    """
    pass  # todo(loya) load subjects task by folder structures.


# todo(kess) add optiong to also include PCA
def train_model(subjects, subjects_task, args):
    """Train a localizer model

    :param subjects: The subjects to train on.
    :param subjects_task: The subjects' task to train on.
    :param args:
    :return: A localizer model
    """
    # todo(kess) this is where you change the model type.
    return Localizer(subjects, subjects_task)


def get_benchmark(localizer, subjects, subjects_task):
    """Get a benchmark of the localizer on the subjects and subjects_task
    2-norm mean.

    :param localizer:
    :param subjects: The subjects to test on.
    :param subjects_task: The subjects' task to test on.
    :return: The benchmark.
    """
    predictions = localizer.predict(subjects)
    return (sum(map(lambda subject_task, prediction: np.linalg.norm(subject_task - prediction),
                         zip(subjects_task, predictions)))).astype(dtype) / len(subjects)


def benchmark(subjects, subjects_task, args):
    benchmark_list = []
    for train_indices, test_indices in sklearn.model_selection.LeaveOneOut().split(subjects):
        localizer = train_model(subjects[train_indices],
                                subjects_task[train_indices],
                                args)
        benchmark_list.append(
            get_benchmark(localizer,
                          subjects[test_indices],
                          subjects_task[test_indices]))

    return np.mean(benchmark_list), np.std(benchmark_list), benchmark_list


def main():
    if ARGS.train and not ARGS.predict:
        validate_train_args(ARGS)
        subjects = load_subjects(ARGS)
        subjects_task = load_subjects_task(ARGS)

        if ARGS.benchmark:
            mean, std, raw = benchmark(subjects, subjects_task, ARGS)
            print(f"Benchmark Results:")
            print(f"Mean: {mean}")
            print(f"STD: {std}")
            print(f"Raw data: {raw}")

        train_model(subjects, subjects_task, ARGS).save_to_file(ARGS.output_file)

    elif ARGS.predict and not ARGS.train:
        validate_predict_args(ARGS)
        subjects = load_subjects(ARGS)
        localizer = Localizer.load_from_file(ARGS.model_file)
        predictions = localizer.predict(subjects)
        utils.utils.create_dir(ARGS.output_dir)
        for subject, prediction in zip(subjects, predictions):
            subject_result_file = os.path.join(ARGS.output_dir, f'{subject.name}_result.pcl')
            pickle.dump(open(subject_result_file, 'wb'), prediction)
    else:
        PARSER.print_help()
        raise ValueError("Either --train or --predict must be provided, and not both.")


if __name__ == '__main__':
    main()
