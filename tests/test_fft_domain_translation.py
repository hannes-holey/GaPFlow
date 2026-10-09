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
import math

import numpy as np
import pytest

from GaPFlow.parallel import DomainDecomposition, FFTDomainTranslation, MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def _nx_ny_splits(n):
    nx = int(math.floor(math.sqrt(n)))
    return nx, n // nx


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


def test_fft_grid_size_periodicity():
    """Verify FFT grid sizes are set correctly based on periodicity."""
    grid_pp = make_grid(16, 16, True, True)
    fft_pp = FFTDomainTranslation(DomainDecomposition(grid_pp))
    assert fft_pp.Nx_fft == 16 and fft_pp.Ny_fft == 16

    grid_pd = make_grid(16, 16, True, False)
    fft_pd = FFTDomainTranslation(DomainDecomposition(grid_pd))
    assert fft_pd.Nx_fft == 16 and fft_pd.Ny_fft == 31

    grid_dp = make_grid(16, 16, False, True)
    fft_dp = FFTDomainTranslation(DomainDecomposition(grid_dp))
    assert fft_dp.Nx_fft == 31 and fft_dp.Ny_fft == 16

    grid_dd = make_grid(16, 16, False, False)
    fft_dd = FFTDomainTranslation(DomainDecomposition(grid_dd))
    assert fft_dd.Nx_fft == 32 and fft_dd.Ny_fft == 32


# ---------------------------------------------------------------------------
# Multi-rank tests — correct behaviour requires nx_splits > 1 (e.g. 4 ranks)
# ---------------------------------------------------------------------------

# Grid sizes that are divisible by the expected splits for common rank counts.
# 16x16 works for 1, 2, 4 ranks; 8x8 for 1, 2, 4.
BLOCK_GRID_CASES = [
    (16, 16, True, True),
    (16, 16, True, False),
    (16, 16, False, True),
    (16, 16, False, False),
    (8, 8, False, False),
    # Divisible by 3 in both axes — exercises 3-rank (1x3) and 6-rank (2x3) splits
    (12, 12, True, True),
    (12, 12, False, False),
    (18, 12, False, False),
]


@pytest.mark.parametrize("Nx,Ny,px,py", BLOCK_GRID_CASES)
def test_exchange_plan_x_offsets(Nx, Ny, px, py):
    """recv_map entries must carry the sender's X offset, not the global Nx.

    This catches the old bug where buffer sizes were hardcoded to (self.Nx, ...).
    Each recv entry's x_size must equal the sender's local subdomain width, and
    x_start + x_size must not exceed Nx.
    """
    grid = make_grid(Nx, Ny, px, py)
    decomp = DomainDecomposition(grid)
    fft_trans = FFTDomainTranslation(decomp)

    for other_rank, info in fft_trans.recv_map.items():
        assert info['x_size'] <= Nx, \
            f"rank {rank}: recv from {other_rank}: x_size {info['x_size']} > Nx {Nx}"
        assert info['x_start'] + info['x_size'] <= Nx, \
            f"rank {rank}: recv from {other_rank}: x_start+x_size overflows Nx"
        # With block decomp and nx_splits > 1, no single sender covers full X
        nx_splits, _ = _nx_ny_splits(size)
        if nx_splits > 1:
            assert info['x_size'] < Nx, \
                f"rank {rank}: recv from {other_rank}: x_size == Nx implies stripe, not block"


@pytest.mark.parametrize("Nx,Ny,px,py", BLOCK_GRID_CASES)
def test_embed_global_content(Nx, Ny, px, py):
    """After embed, gathering the FFT buffer across all ranks must reproduce
    the original global field at the correct X/Y positions.

    For non-periodic directions, rows/columns beyond Nx/Ny must be zero (padding).
    """
    grid = make_grid(Nx, Ny, px, py)
    decomp = DomainDecomposition(grid)
    fft_trans = FFTDomainTranslation(decomp)

    # Build a global reference field: value at (i, j) = i * Ny + j (unique per cell)
    x0 = decomp.subdomain_locations[0]
    y0 = decomp.subdomain_locations[1]
    nx_loc, ny_loc = decomp.nb_subdomain_grid_pts
    src = np.array([[(x0 + i) * Ny + (y0 + j)
                     for j in range(ny_loc)]
                    for i in range(nx_loc)], dtype=float)

    fft_Nx_loc, fft_Ny_loc = fft_trans.fft_engine.nb_subdomain_grid_pts
    fft_buf = np.zeros((fft_Nx_loc, fft_Ny_loc), dtype=float)
    fft_trans.embed(src, fft_buf)

    # Gather FFT buffers to rank 0 and reconstruct global FFT field
    fft_y0 = fft_trans.fft_engine.subdomain_locations[1]
    all_bufs = comm.gather(fft_buf, root=0)
    all_y0 = comm.gather(fft_y0, root=0)
    all_ny = comm.gather(fft_Ny_loc, root=0)

    if rank == 0:
        global_fft = np.zeros((fft_trans.Nx_fft, fft_trans.Ny_fft), dtype=float)
        for buf, gy0, gny in zip(all_bufs, all_y0, all_ny):
            # FFT engine owns full X — buf has shape (Nx_fft, gny)
            global_fft[:, gy0:gy0 + gny] = buf

        # Within the physical domain [0:Nx, 0:Ny], values must match i*Ny+j
        for i in range(Nx):
            for j in range(Ny):
                expected = i * Ny + j
                assert global_fft[i, j] == expected, \
                    f"global_fft[{i},{j}] = {global_fft[i, j]}, expected {expected}"

        # Padding rows/cols must be zero
        if not px:
            assert np.all(global_fft[Nx:, :] == 0.0), "X-padding rows not zero"
        if not py:
            assert np.all(global_fft[:, Ny:] == 0.0), "Y-padding cols not zero"


@pytest.mark.parametrize("Nx,Ny,px,py", BLOCK_GRID_CASES)
def test_roundtrip_global_consistency(Nx, Ny, px, py):
    """After embed+extract, each rank must recover exactly its original block —
    not just any data, but the correct data at the correct position.

    This is the definitive correctness test: it verifies that block-decomposed
    data is correctly scattered to FFT stripes and gathered back.
    """
    grid = make_grid(Nx, Ny, px, py)
    decomp = DomainDecomposition(grid)
    fft_trans = FFTDomainTranslation(decomp)

    # Use globally unique values so any misplacement is detectable
    x0 = decomp.subdomain_locations[0]
    y0 = decomp.subdomain_locations[1]
    nx_loc, ny_loc = decomp.nb_subdomain_grid_pts
    src = np.array([[(x0 + i) * Ny + (y0 + j)
                     for j in range(ny_loc)]
                    for i in range(nx_loc)], dtype=float)

    fft_Nx_loc, fft_Ny_loc = fft_trans.fft_engine.nb_subdomain_grid_pts
    fft_buf = np.zeros((fft_Nx_loc, fft_Ny_loc), dtype=float)
    fft_trans.embed(src, fft_buf)

    dst = np.zeros_like(src)
    fft_trans.extract(fft_buf, dst)

    np.testing.assert_array_equal(
        dst, src,
        err_msg=f"rank {rank}: roundtrip data mismatch at x0={x0}, y0={y0}"
    )
