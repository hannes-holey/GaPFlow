import pathlib
import runpy
import pytest
import subprocess
from GaPFlow.problem import Problem


basedir = pathlib.Path(__file__,'..',).resolve()
# print(scripts)
scripts = list((basedir / 'config').glob('*.yaml'))

@pytest.mark.parametrize('script', scripts, ids=[s.name for s in scripts])
def test_config_file_execution( script):
    myProblem = Problem.from_yaml(str(script))
    myProblem.run()
