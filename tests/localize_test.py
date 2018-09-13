import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from neuralocalize import localize

args = localize.PARSER.parse_args(
    '''
    --benchmark
    --use_task_filename_as_template
    --number_of_tasks 86
    --task_filename AllSubjects_%03d.dtseries.nii
    --input_dir ./test_resources/
    --task_ordered_subjects_filename subjects.txt
    '''.split()
)

localize.main(args)
