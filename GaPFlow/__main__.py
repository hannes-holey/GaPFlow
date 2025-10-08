from GaPFlow.problem import Problem
from argparse import ArgumentParser
# from jaxtyping import install_import_hook
# install_import_hook("GaPFlow", "beartype")


def get_parser():

    parser = ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input',
                          dest="filename",
                          help="YAML input file",
                          required=True)

    return parser


if __name__ == "__main__":

    # load problem from yaml
    parser = get_parser()
    args = parser.parse_args()
    problem = Problem.from_yaml(args.filename)

    # Run
    # maybe pass some args to run method as they are not problem specific
    # e.g., max_steps, tolerance, adaptive, integrator (only MC a.t.m.)
    problem.run()
