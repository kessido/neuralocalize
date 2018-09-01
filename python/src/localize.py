#!/usr/bin/python

import argparse
import os

# TODO(loya) add full description.
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
                    ''') # TODO(loya) decide on a format.
PARSER.add_argument('--predict', action='store_true', )
PARSER.add_argument('--input_dir', default='./',
                    help='''The path to the input file(s).
                    In training mode, the input files should include subject rest data and task data in the
                    following format:
                    <TODO(loya) format>.
                    In predict mode, only the rest files are required.
                    ''') # TODO(loya) separate to rest and task
PARSER.add_argument('--output_dir', default='./',
                    help='For training mode. The directory where the output files will be written to.')
PARSER.add_argument('--model_dir', default='./',
                    help='Required for predict mode. The directory where the trained model is located.')

ARGS = PARSER.parse_args()


def validate_predict_args(args):
    """Validates the prediction arguments are correct.
    :param args:
    :return:
    """
    if not os.path.exists(args.input_dir):
        raise ValueError("Input file doesn't exist.")

def validate_train_args(args):
    if not os.path.exists(args.input_dir):
        raise ValueError("Input file doesn't exist.")

def main():
    if ARGS.train and not ARGS.predict:
        validate_train_args(ARGS)
        # TODO(loya) call the train methods.
    elif ARGS.predict and not ARGS.train:
        validate_predict_args(ARGS)
        # TODO(loya) call predict.
    else:
        PARSER.print_help()
        raise ValueError("Either --train or --predict must be provided, and not both.")


if __name__ == '__main__':
    main()
