from hans_mugrid.problem import Problem

if __name__ == "__main__":

    problem = Problem.from_yaml('journal.yaml')
    problem.run()
