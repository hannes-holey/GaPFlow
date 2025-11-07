#
# Copyright 2025 Hannes Holey
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
import pytest

from GaPFlow.md.moltemplate import _get_MPI_grid, _read_coords_from_lt


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
