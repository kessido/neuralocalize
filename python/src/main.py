import argparse
import os

# TODO(loya) add full description.
PARSER = argparse.ArgumentParser(description="Description of the module")

# TODO(loya) extend. Add file types etc.
PARSER.add_argument('--input_file', default='./', help='The path to the input file.')
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