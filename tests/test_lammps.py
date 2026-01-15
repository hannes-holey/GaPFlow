#
# Copyright 2026 Hannes Holey
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
import pytest

from GaPFlow.md._lammps import lammps
from GaPFlow.md.runner import PARALLEL


def show_info(lmp):
    print()
    print('OS:', lmp.get_os_info())
    print('Shared lib: ', lmp.lib._name)
    print('LAMMPS Version: ', lmp.version())
    print('MPI: ', lmp.has_mpi_support)
    print('mpi4py: ', lmp.has_mpi4py)
    print('packages: ', lmp.installed_packages)


@pytest.mark.skipif(not PARALLEL, reason="Evaluate only for parallel implementations")
def test_lammps_parallel():

    lmp = lammps.lammps(name='mpi', cmdargs=['-log', 'none', "-screen", 'none'])

    show_info(lmp)

    assert lmp.has_mpi_support
    assert lmp.has_mpi4py
    assert 'MANYBODY' in lmp.installed_packages
    assert 'MOLECULE' in lmp.installed_packages
    assert 'EXTRA-FIX' in lmp.installed_packages

    lmp.close()


def test_lammps_serial():

    lmp = lammps.lammps(name='mpi', cmdargs=['-log', 'none', "-screen", 'none'])

    show_info(lmp)

    assert 'MANYBODY' in lmp.installed_packages
    assert 'MOLECULE' in lmp.installed_packages
    assert 'EXTRA-FIX' in lmp.installed_packages

    lmp.close()
