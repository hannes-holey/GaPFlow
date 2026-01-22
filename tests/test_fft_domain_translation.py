#
# Copyright 2025 Christoph Huber
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
"""
Tests for FFTDomainTranslation class.

Run with: mpirun -np 1 python -m pytest tests/test_fft_domain_translation.py -v
          mpirun -np 2 python -m pytest tests/test_fft_domain_translation.py -v
          mpirun -np 4 python -m pytest tests/test_fft_domain_translation.py -v
"""
import numpy as np
import pytest
from mpi4py import MPI

from GaPFlow.parallel import DomainDecomposition, FFTDomainTranslation


def make_grid(Nx, Ny, periodic_x, periodic_y):
    """Create minimal grid config for DomainDecomposition."""
    bc_type_x = 'P' if periodic_x else 'D'
    bc_type_y = 'P' if periodic_y else 'D'
    return {
        'Nx': Nx, 'Ny': Ny,
        'bc_xW': [bc_type_x] * 3,
        'bc_xE': [bc_type_x] * 3,
        'bc_yS': [bc_type_y] * 3,
        'bc_yN': [bc_type_y] * 3,
        'bc_xW_D_val': 0, 'bc_xE_D_val': 0,
        'bc_yS_D_val': 0, 'bc_yN_D_val': 0,
    }


# Test parameters: (Nx, Ny, periodic_x, periodic_y, expected_Nx_fft, expected_Ny_fft)
GRID_CASES = [
    # Even grid
    (16, 16, True, True, 16, 16),          # Fully periodic
    (16, 16, True, False, 16, 31),         # Semi-periodic (y free)
    (16, 16, False, True, 31, 16),         # Semi-periodic (x free)
    (16, 16, False, False, 32, 32),        # Fully non-periodic
    # Uneven grid
    (17, 13, True, True, 17, 13),
    (17, 13, True, False, 17, 25),
    (17, 13, False, True, 33, 13),
    (17, 13, False, False, 34, 26),
    # Rectangular grid
    (32, 16, True, True, 32, 16),
    (32, 16, False, False, 64, 32),
]


@pytest.mark.parametrize("Nx,Ny,px,py,Nx_fft_exp,Ny_fft_exp", GRID_CASES)
def test_fft_grid_size(Nx, Ny, px, py, Nx_fft_exp, Ny_fft_exp):
    """Verify FFT grid size matches expected values for each periodicity."""
    grid = make_grid(Nx, Ny, px, py)
    decomp = DomainDecomposition(grid)
    fft_trans = FFTDomainTranslation(decomp)

    assert fft_trans.Nx_fft == Nx_fft_exp, f"Nx_fft: {fft_trans.Nx_fft} != {Nx_fft_exp}"
    assert fft_trans.Ny_fft == Ny_fft_exp, f"Ny_fft: {fft_trans.Ny_fft} != {Ny_fft_exp}"
    assert tuple(fft_trans.fft_engine.nb_domain_grid_pts) == (Nx_fft_exp, Ny_fft_exp)


@pytest.mark.parametrize("Nx,Ny,px,py,_,__", GRID_CASES)
def test_roundtrip(Nx, Ny, px, py, _, __):
    """Verify embed followed by extract recovers original data."""
    comm = MPI.COMM_WORLD
    grid = make_grid(Nx, Ny, px, py)
    decomp = DomainDecomposition(grid)
    fft_trans = FFTDomainTranslation(decomp)

    # Create test data on GaPFlow domain (local subdomain)
    local_Nx = decomp.nb_subdomain_grid_pts[0]
    local_Ny = decomp.nb_subdomain_grid_pts[1]
    np.random.seed(42 + comm.rank)
    src = np.random.rand(local_Nx, local_Ny)

    # Allocate FFT domain buffer
    fft_local_Nx = fft_trans.fft_engine.nb_subdomain_grid_pts[0]
    fft_local_Ny = fft_trans.fft_engine.nb_subdomain_grid_pts[1]
    fft_buf = np.zeros((fft_local_Nx, fft_local_Ny), dtype=src.dtype)

    # Embed: GaPFlow -> FFT
    fft_trans.embed(src, fft_buf)

    # Extract: FFT -> GaPFlow
    dst = np.zeros_like(src)
    fft_trans.extract(fft_buf, dst)

    # Verify roundtrip
    np.testing.assert_allclose(dst, src, rtol=1e-14,
                               err_msg=f"Roundtrip failed for px={px}, py={py}")


def test_needs_redistribution_flag():
    """Verify _needs_redistribution is set correctly."""
    # Fully periodic should not need redistribution
    grid_pp = make_grid(16, 16, True, True)
    decomp_pp = DomainDecomposition(grid_pp)
    fft_pp = FFTDomainTranslation(decomp_pp)
    assert not fft_pp._needs_redistribution

    # Any non-periodic direction needs redistribution
    for px, py in [(True, False), (False, True), (False, False)]:
        grid = make_grid(16, 16, px, py)
        decomp = DomainDecomposition(grid)
        fft = FFTDomainTranslation(decomp)
        assert fft._needs_redistribution, f"Expected redistribution for px={px}, py={py}"
