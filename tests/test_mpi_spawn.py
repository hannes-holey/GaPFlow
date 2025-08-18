#
# Copyright 2024-2025 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import sys
import subprocess
import pytest


@pytest.fixture(scope="session", params=[2, 4, 6, 8])
def command(request):

    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spawner_nomerge.py')
    command = f'mpirun -n 1 --oversubscribe {sys.executable} {fname} {request.param}'

    yield command


# @pytest.mark.skip(reason='Broadcasting with spawned processes stalls on some systems.')
def test_spawn(command):

    try:
        subprocess.run(command.split(), check=True, stderr=True, env=os.environ)
    except subprocess.CalledProcessError:
        assert False
