#!/usr/bin/python

import argparse
import os

# TODO(loya) add full description.
PARSER = argparse.ArgumentParser(description=("""
A module for the Localize model.
The model predicts individual differences in brain activity.

Given an input of rest fMRI data and tasks (TODO(loya) elaborate),
the model outputs the prediction of the activation in this task into the output directory.

For more information, see Tavor et al. Task-free MRI predicts individual differences
in brain activity during task performance:
http://science.sciencemag.org/content/352/6282/216"""
                                              ))

# TODO(loya) extend. Add file types etc.
PARSER.add_argument('--input_file', default='./', help='The path to the input file.')
PARSER.add_argument('--task', default='./', help='The path to the input file.') # TODO(loya) ???
PARSER.add_argument('--output_dir', default='./', help='The directory where the output files will be written to.')

ARGS = PARSER.parse_args()

def validate_args(args):
    if not os.path.exists(args.input_file):
        raise ValueError("Input file doesn't exist.")



def main():
    # TODO(loya) validate the actions we take and run them here.
    pass

if __name__ == '__main__':
    main()