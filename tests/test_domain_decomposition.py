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


