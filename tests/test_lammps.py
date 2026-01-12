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
