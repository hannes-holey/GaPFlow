import io
import pytest
import numpy as np

from GaPFlow.models.continuum_wall_stress import NewtonParabolicFlowFluidGap, SlipLengthSlip, Operations

from GaPFlow.io import read_yaml_input
from GaPFlow import Problem
from GaPFlow.config_utils import build_config
import matplotlib.pyplot as plt

from GaPFlow.viz import evaluate_quadfield_on_nodes


@pytest.mark.parametrize('Ls', [0, 0.1] )
def test_couette_symmetry(Ls, tmp_path):

    # import tol_colors as tc

    config = """
options:
    output: data/journal
    write_freq: 1000
    silent: False
grid:
    Lx: 1.
    Ly: 1.
    Nx: 32
    Ny: 1
    xE: ['P', 'P', 'P',]
    xW: ['P', 'P', 'P',]
    yS: ['P', 'P', 'P',]
    yN: ['P', 'P', 'P',]
geometry:
    type: journal
    hmin: 1.
    hmax: 1.
    U_top: 1.
    V_top: 0.
    U_bot: -1.
    V_bot: 0.
numerics:
    solver: implicit
    dt: 1.0
    tol: 1e-13
    max_it: 10000

properties:
    EOS: DH
    rho0: 1.
    P0: 0.001 # the pressure unit is viscosity * velocity / hmin # Here we set this value to 1. as anb example, but we will update later to give physically more meaningful values.
    C1: 345423.1433506045  # Which is the classical valye 3.5e10 / the classical P0 value of 101325
    C2: 1.23
    shear: 1. # This together with the velocity determines the pressure unit
    bulk: 0.0

implicit_solver:
    linear_solver: direct
    scaling: False
    newton_relax: 0.5
    p_init: 1.
    equations:
        cavitation: False
    max_iter: 100000
    """


    models = dict(
    wall_stress = NewtonParabolicFlowFluidGap(mu0=1.) ,
    operations = Operations()
    )

    if Ls > 0:
        models.update(**dict(
        slip_top = SlipLengthSlip(Ls, 1),
        slip_bot = SlipLengthSlip(Ls, 1)))
        slip = True
    else:
        slip = False

    config_hans_slip = build_config(config, **{'implicit_solver.physics.slip': slip,
                                               'options.output': str(tmp_path)
                                               })

    with io.StringIO(config_hans_slip) as ymlfile:
        inp = read_yaml_input(ymlfile)



    problem = Problem(options=inp['options'],
                       grid=inp['grid'],
                       numerics=inp['numerics'],
                       prop=inp['properties'],
                       geo=inp['geometry'],
                       fem_solver=inp['fem_solver'],
                       energy_spec=inp['energy_spec'],
                       custom_models=models
                      )

    problem.run()

    np.testing.assert_allclose(evaluate_quadfield_on_nodes(problem, 'Uf_top'), -evaluate_quadfield_on_nodes(problem, 'Uf_bot'), atol=1e-10)
    np.testing.assert_allclose(evaluate_quadfield_on_nodes(problem, 'Vf_top'), 0, atol=1e-10)
    np.testing.assert_allclose(evaluate_quadfield_on_nodes(problem, 'Uf_top'), 0, atol=1e-10)
    if slip:
        np.testing.assert_allclose(evaluate_quadfield_on_nodes(problem, 'Us_top'), evaluate_quadfield_on_nodes(problem, 'Us_bot'), atol=1e-10)
        np.testing.assert_allclose(evaluate_quadfield_on_nodes(problem, 'Vs_top'), 0, atol=1e-10)
        np.testing.assert_allclose(evaluate_quadfield_on_nodes(problem, 'Us_top'), 0, atol=1e-10)
