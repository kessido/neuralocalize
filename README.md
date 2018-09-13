This package implements the model training and prediction described in Tavor el al,
Task-free MRI predicts individual differences in brain activity during task performance:
http://science.sciencemag.org/content/352/6282/216.

This is the git repository for the seminar of Ido Tavor's team of 2018 - Ido Kessler, Noam Loya and Itay Itzhak

# Neural Localizer


# Installation

A simple pip install should work
```
pip install git+https://github.com/kessido/Neuroscience-seminar
```

# Usage (Python)

## 0. Loading the data:
```
import neuralocalize
import cifti
import numpy as np

# Loading the files
subjects = []
subjects_info = [('100307','path/to/100307/MNINonLinear/Results/'),...,('101915','path/to/101915/MNINonLinear/Results/')]
for subject_name, subject_path in subjects_info:
	subject = neuralocalize.Subject(subject_name)
	subject.load_from_directory(subject_path)
	subjects.append(subject)

subjects = np.array(subjects)
tasks = np.array(cifti.read('path/to/subjects/tasks/file')[0])
train_indexes = range(0,5)
test_indexes = range(5,10)
```

## 1. Standard usage:
```
# Predicting
localizer_model = neuralocalize.Localizer() # Load the precomputed PCA results
localizer_model.fit(subjects[train_indexes], tasks[train_indexes])
prediction = localizer_model.predict(subjects[test_indexes])

print(np.linalg.norm(prediction-tasks[test_indexes]))
```

## 2. Use your pca:
```
pca_results = np.array(cifti.read('path/to/pca_result')[0])
localizer_model = neuralocalize.Localizer(pca_result=pca_result)
```

## 3. Compute new PCA(Dont use - very long time):
```
localizer_model = neuralocalize.Localizer(subjects=subjects, compute_pca=True)
```


## 4. Predict using your model:
```
class SimplePredictorGenerator:
	class PredictorModel:
		def __init__(self, beta):
			self._beta = beta
		
		def predict(self, X):
			return X @ self._beta

	def fit(self, X, y):
		"""Must return an object that has a predict method."""
		betas = [sl.lstsq(subject_feature, task)[0] for subject_feature, task in zip(X, y)]
		return SimplePredictorGenerator.PredictorModel(np.mean(np.array(betas), axis=0))

# Predicting
my_predictor_generator = SimplePredictorGenerator()
localizer_model = neuralocalize.Localizer(predictor_generator=my_predictor_generator)
localizer_model.fit(subjects[train_indexes], tasks[train_indexes])
prediction = localizer_model.predict(subjects[test_indexes])

print(np.linalg.norm(prediction-tasks[test_indexes]))
```


# Usage (Command line)

## 0. Input dir structure
The input dir structure is taken from the human connectome project.

```
input_dir
|
|
|__ Subjects
|     |
|     |__ {SubjectName} # i.e 100307
|     |     |__ MNINonLinear
|     |           |__ Results
|     |                 |__ rfMRI_REST1_LR
|     |                 |     |__ rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
|     |                 |__ rfMRI_REST1_RL
|     |                 |     |__ rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii
|     |                 |__ rfMRI_REST2_LR
|     |                 |     |__ rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
|     |                 |__ rfMRI_REST2_RL
|     |                       |__ rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii
|     |__ {SubjectName}
|     |     |__ ...
|     |__ {SubjectName}
|     |     |__ ...
|     |__ ...
|
|__ Tasks
|     |__ {Tasks files .dtseries.nii}
|
|__ {Subject ordering inside the tasks files .txt}

```

Notice that the "Subject ordering inside the tasks files", is just a simple text file listing the subjects by their appearance in the tasks files.


## 1. Training the model

```
localize --train --task_filename AllSubjects_001.dtseries.nii --input_dir ./MyTrainingInputFolder/ --output_dir ./MyTestingInputFolder/ --model_filename Task001Model.pcl.gz --task_ordered_subjects_filename subjects.txt
``` 

This will output a train model over all subjects in the input dir.

## 2. Predicting with the model

```
localize --predict --task_filename AllSubjects_001.dtseries.nii --input_dir ./MyTestingInputFolder/ --output_dir ./TasksPrediction/ --model_filename Task001Model.pcl.gz --prediction_results_filename task001.dtseries.nii
``` 

This will output a "task001.dtseries.nii" file into the "TasksPrediction" folder, containing the predictions of the subjects in the input_dir on the specific task.

## 3. Benchmarking 

```
localize --benchmark --task_filename AllSubjects_%03d.dtseries.nii --number_of_tasks 86 --prediction_results_filename task%03d_benchmark_leave_one_out_prediction.dtseries.nii --output_dir ./BenchmarkingPredictionResults/ --input_dir ./MyInputDir/ --task_ordered_subjects_filename subjects.txt --input_dir ./MyTestingInputFolder/
``` 

This will output 86 files of predictions inside "BenchmarkingPredictionResults" folder. 
Each result file containes all the subjects prediction when training the model on everyone but them on the specific task. (Leave one out).

