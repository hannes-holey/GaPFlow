import pytest

from GaPFlow.parallel import DomainDecomposition
import numpy as np


def test_1d_grid_yy_vals():
    """
    Test that yy values at ghost celles are correct in the 1D case
    :return:
    """
    nx = 3
    grid = {'Nx': nx,
         'Lx': 2.,
         'dx': 0.5,
         'Ny': 1,
         'Ly': 1.0,
         'dy': 1.0,
         'dim': 1,
         'bc_xW': ['P', 'P', 'P'],
         'bc_xE': ['P', 'P', 'P'],
         'bc_yS': ['P', 'P', 'P'],
         'bc_yN': ['P', 'P', 'P']}

    numerics = {'solver': 'explicit',}

    decomp = DomainDecomposition(grid, numerics)

    np.testing.assert_allclose(decomp.yy, np.array([-0.5, 0.5, 1.5]).reshape((1,-1)) * np.ones([nx+2, 1]))


@pytest.mark.parametrize("bc_dict", [
    {'bc_xW': ['D', 'D', 'D'],
     'bc_xE': ['D', 'D', 'D'],
     'bc_yS': ['P', 'P', 'P'],
     'bc_yN': ['P', 'P', 'P']},
    {'bc_xW': ['P', 'P', 'P'],
     'bc_xE': ['P', 'P', 'P'],
     'bc_yS': ['P', 'P', 'P'],
     'bc_yN': ['P', 'P', 'P']},
    {'bc_xW': ['P', 'P', 'P'],
     'bc_xE': ['P', 'P', 'P'],
     'bc_yS': ['D', 'D', 'D'],
     'bc_yN': ['D', 'D', 'D']},
    {'bc_xW': ['D', 'D', 'D'],
     'bc_xE': ['D', 'D', 'D'],
     'bc_yS': ['D', 'D', 'D'],
     'bc_yN': ['D', 'D', 'D']}
])
def test_grid_vals_periodic(bc_dict):
    """
    Test that yy values at ghost celles are correct in the 1D case
    :return:
    """
    nx = 3
    ny = 3
    grid = {'Nx': nx,
         'Lx': nx * 1.,
         'dx': 1.,
         'Ly': ny * 1.,
         'Ny': ny,
         'dy': 1.0,
         'dim': 1,
         }

    grid = grid | bc_dict

    numerics = {'solver': 'explicit',}

    decomp = DomainDecomposition(grid, numerics)

    np.testing.assert_allclose(decomp.yy, np.array([-0.5, 0.5, 1.5, 2.5, 3.5]).reshape((1,-1)) * np.ones([nx+2, 1]))
    np.testing.assert_allclose(decomp.xx, np.array([-0.5, 0.5, 1.5, 2.5, 3.5]).reshape((-1,1)) * np.ones([1, ny+2]))


