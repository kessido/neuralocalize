#!/usr/bin/python

import argparse
import os

import numpy as np
import sklearn

from . import prediction, utils
from .utils import constants

PARSER = argparse.ArgumentParser(description=
"""
A module for the Localize model.
The model predicts individual differences in brain activity.

This module has two main functionalities:
1. Training a model: Given a task and subject data (rest and task data), trains a model to predict the brain activity in
this task given rest fMRI data.
2. Loading a trained model, and use it to predict task brain activity.

For more information about the model, see Tavor et al. Task-free MRI predicts individual differences
in brain activity during task performance:
http://science.sciencemag.org/content/352/6282/216""")

# todo(kess) multiple task prediction.
PARSER.add_argument('--train', action='store_true', help=
'''
Training mode.
Given training data - subject rest and task, trains a model and outputs it into the output directory.
''')

PARSER.add_argument('--predict', action='store_true', help=
'''
Predict mode.
Given a trained model, and subjects rest-fMRI, predict the subjects' activation when preforming a certain task the model was trained on.
''')


PARSER.add_argument('--benchmark', action='store_true', help=
'''
If presented, a Leave One Out testing method will be used, where the 
model will be trained each time excluding one subject, and then testing the model on this subject.
The mean prediction accuracy, and std, will then be presented in the console. 
The prediction for all the tasks will later be saved to the output_dir.
''')

PARSER.add_argument('--input_dir', default='./', help=
'''
The path to the input directory.
In training mode, the input files should include subject rest data and task data in the HCP format:
The session files are in stored as follows:
base_dir | {subject_name} | MNINonLinear | Results | rfMRI_REST{session_num}_{sides} | rfMRI_REST{session_id}_{sides}_Atlas_MSMAll_hp2000_clean.dtseries.nii
The task files are stored as follows:
base_dir | Tasks | task_filename
In predict mode, only the rest files are required.
''')

PARSER.add_argument('--output_dir', default='./results',  help=
'''
Required for prediction and benchmark mode. 
The output folder to use for outputting the predictions.
The output will later be saved in the following way:
output_dir | {prediction_results_filename}
if benchmarking it will be:
output_dir | AllSubjects_{task_number}_result.dtseries.nii
where the subject will be saved as a time-series.
''')

PARSER.add_argument('--compute_pca', action='store_true', help='Defaults to False. Set to True of you want to run the pca. This takes time. Is not relevant for prediction.')
PARSER.add_argument('--pca_result_filename', default=None, help='Optional. Load PCA result from this location. Is not relevant for prediction.')

PARSER.add_argument('--task_filename', help=
'''
Name of the task files. Stored in {input_dir}/Tasks. Is not relevant for prediction.
In benchmark, run over the range of 1 to nunmber_of_tasks to load different tasks 
using the task_filename as a template. In pseudocode: ['/'.join[output_dir,'Tasks',task_filename].format[i+1] for i in range[number_of_tasks]]
Is not relevant for prediction.''')

PARSER.add_argument('--number_of_tasks_for_benchmark', default=86, type=int, help='Defaults to 86. number of task to load. Is relevant only for benchmark.')

PARSER.add_argument('--task_ordered_subjects_filename', default='subjects.txt', help=
'''
The file name for the file holding a list of the subject ids in the order they appear in the task matrices.
Should be located under: input_dir | task_ordered_subjects_filename
Is not relevant for prediction.
''')

PARSER.add_argument('--model_filename', default='model.pcl.gz', help=
'''
In Prediction mode: It's the file containing the trained localizer model.
Should be under: input_dir | model_filename
In Train mode: It's the output file to save the model to: output_dir | model_filename
''')

PARSER.add_argument('--prediction_results_filename', default='result.dtseries.nii', help=
'''
When predicting a task, this is the output filename that will be used.''')

PARSER.add_argument('--number_of_subjects', default=0, type=int, help=
"""
Optional. If given, take the first number_of_subjects instead of all the subject under the directory.
""")


ARGS = PARSER.parse_args()


def validate_predict_args(args):
	if not os.path.exists(args.input_dir):
		raise ValueError("Input folder doesn't exist.")

	if not os.path.exists(os.path.join(args.input_dir, args.model_filename)):
		raise ValueError("Model file doesn't exist.")

def validate_train_and_benchmark_args(args):
	if not os.path.exists(args.input_dir):
		raise ValueError("Input folder doesn't exist.")
	
	if args.task_filename is None:
		raise ValueError("Task filename was not provided.")
	

def validate_args(args):
	if sum([args.train, args.predict, args.benchmark]) != 1:
		PARSER.print_help()
		raise ValueError("Either --train or --predict or --benchmark must be provided, and not both.")

def load_subjects(input_dir, number_of_subjects):
	subjects = []
	subj_dir = os.path.join(input_dir, 'Subjects')
	subject_folders = os.listdir(subj_dir)
	if number_of_subjects > 0 and number_of_subjects < len(subject_folders):
		subject_folders = subject_folders[:number_of_subjects]
	print("Loading", len(subject_folders), "subjects from", subj_dir)
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


def arrange_task_by_subject(subjects, subj_index_dict, all_subjects_task):
	tasks_ordered_by_subj = []
	for subj in subjects:
		tasks_ordered_by_subj.append(all_subjects_task[subj_index_dict[subj.name]])
	return np.array(tasks_ordered_by_subj)


def load_subjects_task(args, subjects):
	full_path_to_ordered_subjs = os.path.join(args.input_dir, args.task_ordered_subjects_filename)
	subj_index_dict = _get_ordered_subjects_list(full_path_to_ordered_subjs)

	full_path_to_tasks = [os.path.join(args.input_dir, 'Tasks', args.task_filename)]
	if args.benchmark:
		full_path_to_tasks = [full_path_to_tasks[0] % (i + 1) for i in range(args.number_of_tasks)]
	
	for subject_task_path in full_path_to_tasks:
		task, _ = utils.cifti_utils.load_cifti_brain_data_from_file(subject_task_path)
		res = arrange_task_by_subject(subjects, subj_index_dict, task)
		yield res

def get_benchmark(localizer, subjects, subjects_task):
	"""Get a benchmark of the localizer on the subjects and subjects_task
	2-norm mean.
	:return: The benchmark.
	"""
	predictions = localizer.predict(subjects)
	res = []
	for subject_task, prediction in zip(subjects_task, predictions):
		dif = np.abs(np.array(subject_task, dtype=constants.DTYPE) - np.array(prediction, dtype=constants.DTYPE))
		res.append(np.linalg.norm(dif))
	return np.mean(np.array(res, dtype=constants.DTYPE)), predictions


def benchmark_single_task(subjects, subjects_task, localizer):
	predictions = None
	benchmark_list = []
	subjects_task = np.array(subjects_task, dtype=constants.DTYPE)
	subjects = np.array(subjects)
	for train_indices, test_indices in sklearn.model_selection.LeaveOneOut().split(subjects):
		print("Run on ", train_indices, test_indices)
		localizer.fit(subjects[train_indices],
					  subjects_task[train_indices])
		norm_mean, predictions_for_add = get_benchmark(localizer,
													   subjects[test_indices],
													   subjects_task[test_indices])
		benchmark_list.append(norm_mean)
		if predictions is None:
			predictions = predictions_for_add
		else:
			predictions = np.concatenate([predictions, predictions_for_add], axis=0)
	return np.mean(benchmark_list), np.std(benchmark_list), np.max(benchmark_list), benchmark_list, predictions


def benchmark_tasks(subjects, subjects_tasks, args, localizer):
	print("Benchmark Starting...")
	raws = []

	for i, subjects_task in enumerate(subjects_tasks):
		print("Benchmarking task ", i)
		mean, std, max, raw, predictions = benchmark_single_task(subjects, subjects_task, localizer)
		print("Mean:", mean)
		print("STD:", std)
		print("max:", max)
		raws = np.concatenate([raws, raw])

		prediction_file_name = 'AllSubjects_%03d_results.dtseries.nii' % (i + 1)
		utils.utils.create_dir(args.output_dir)
		prediction_path = os.path.join(args.output_dir, prediction_file_name)
		utils.cifti_utils.save_cifti(predictions, prediction_path)

	return np.mean(raws), np.std(raws), np.max(raws), raws


def main(ARGS=ARGS):
	validate_args(ARGS)
	subjects = load_subjects(ARGS.input_dir, ARGS.number_of_subjects)
		
	if ARGS.train or ARGS.benchmark:
		validate_train_and_benchmark_args(ARGS)
		
		pca_result = None
		if not ARGS.compute_pca and ARGS.pca_result_filename is not None:
			pca_result_path = os.path.join(ARGS.input_dir, ARGS.pca_result_filename)
			pca_result, _ = utils.cifti_utils.load_cifti_brain_data_from_file(pca_result_path)
		
		subjects_tasks = load_subjects_task(ARGS, subjects)
		localizer = prediction.Localizer(subjects=subjects, pca_result=pca_result, compute_pca=ARGS.compute_pca)

		if ARGS.benchmark:
			if len(subjects) <= 1:
				raise ValueError("Not enough subjects to preform leave-one-out.")
			mean, std, max, raw = benchmark_tasks(subjects,
												  subjects_tasks,
												  ARGS,
												  localizer)
			print("Benchmark Results:")
			print("Mean:", mean)
			print("STD:", std)
			print("Max:", max)
			print("Raw data:", raw)

		if ARGS.train:
			output_path = os.path.join(ARGS.output_dir, ARGS.model_filename)
			print("Training Model.")
			localizer.fit(subjects, next(subjects_tasks))
			utils.utils.create_dir(args.output_dir)
			localizer.save_to_file(output_path)

			
	if ARGS.predict:
		validate_predict_args(ARGS)
		localizer = prediction.Localizer.load_from_file(os.path.join(ARGS.input_dir, ARGS.model_filename))
		predictions = localizer.predict(subjects)
		utils.utils.create_dir(ARGS.output_dir)
		utils.cifti_utils.save_cifti(predictions, os.path.join(ARGS.output_dir, ARGS.prediction_results_filename))


if __name__ == '__main__':
	main()
