import os
import pytest

from GaPFlow.moltemplate.init import _get_MPI_grid
from GaPFlow.moltemplate.fluid import _read_coords_from_lt


@pytest.mark.parametrize("Na,size,max_cpu", [(1000, 2, 4),
                                             (30_000, 3, 30),
                                             (30_000, 3, 3),
                                             (100_000, 4, 12),
                                             (500_000, 4, 24),
                                             (1_000_000, 5, 30)
                                             ])
def test_processors_grid(Na, size, max_cpu):

    nx, ny, nz = _get_MPI_grid(Na, size, max_cpu)

    assert nx * ny * nz <= max_cpu


@pytest.mark.parametrize("fname,expected", [('pentane.lt', 5),
                                            ('decane.lt', 10),
                                            ('hexadecane.lt', 16)])
def test_coords_from_lt(fname, expected):

    fname_abspath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 '..', 'examples', 'lmp', 'mol', 'moltemplate_files', fname)

    coords = _read_coords_from_lt(fname_abspath)
    assert coords.shape == (expected, 3)
