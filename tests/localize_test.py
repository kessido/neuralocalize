import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from neuralocalize import localize


def test_pca():
	args = localize.PARSER.parse_args(
		'''
		--benchmark
		--number_of_tasks 10
		--number_of_subjects 2
		--task_filename AllSubjects_%03d.dtseries.nii
		--input_dir ./test_resources/
		--task_ordered_subjects_filename subjects.txt
		--compute_pca
		'''.split()
	)
	
	localize.main(args)

def test_ica():
	args = localize.PARSER.parse_args(
		'''
		--benchmark
		--number_of_tasks 10
		--number_of_subjects 2
		--task_filename AllSubjects_%03d.dtseries.nii
		--input_dir ./test_resources/
		--task_ordered_subjects_filename subjects.txt
		--pca_result_filename GROUP_PCA_rand200_RFMRI.dtseries.nii
		'''.split()
	)
	
	localize.main(args)

	
def test_train_and_predict():
	args = localize.PARSER.parse_args(
		'''
		--train
		--number_of_subjects 1
		--task_filename AllSubjects_001.dtseries.nii
		--input_dir ./test_resources/
		--output_dir ./test_resources/
		--task_ordered_subjects_filename subjects.txt
		'''.split()
	)
	
	localize.main(args)
	
	args = localize.PARSER.parse_args(
		'''
		--predict
		--number_of_subjects 1
		--task_filename AllSubjects_001.dtseries.nii
		--input_dir ./test_resources/
		--prediction_results_filename out.dtseries.nii
		'''.split()
	)
	
	localize.main(args)



def test_benchmarking():
	args = localize.PARSER.parse_args(
		'''
		--benchmark
		--number_of_tasks 10
		--number_of_subjects 2
		--task_filename AllSubjects_%03d.dtseries.nii
		--input_dir ./test_resources/
		--task_ordered_subjects_filename subjects.txt
		'''.split()
	)
	
	localize.main(args)

	
test_train_and_predict()