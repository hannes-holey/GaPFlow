


import io
import pytest
import numpy as np

from GaPFlow.io import read_yaml_input
from GaPFlow import Problem
from GaPFlow.config_utils import build_config
import matplotlib.pyplot as plt
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
    xE: ['P', 'P', 'P']
    xW: ['P', 'P', 'P']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
geometry:
    type: journal
    hmin: 1.
    hmax: 2.
    U: 1.
    V: 0.
numerics:
    solver: implicit
    dt: 1.0
    tol: 1e-3
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
    newton_relax: 0.5
    p_init: 1.
    equations:
        cavitation: False
    max_iter: 100
"""

from GaPFlow.models.continuum_wall_stress import NewtonParabolicFlowFluidGap

gap = NewtonParabolicFlowFluidGap(mu0=1.)

with io.StringIO(config) as ymlfile:
    inp = read_yaml_input(ymlfile)
problem = Problem(options=inp['options'],
                   grid=inp['grid'],
                   numerics=inp['numerics'],
                   prop=inp['properties'],
                   geo=inp['geometry'],
                   fem_solver=inp['fem_solver'],
                   energy_spec=inp['energy_spec'],
                   wall_stress=gap,
                  )

problem.run()

# %%

ref_problem = Problem.from_string(build_config(config, **{'implicit_solver.equations.reynolds': True}))

ref_problem.run()


fig, ax1 = plt.subplots(1, 1 , sharex=True, sharey=False, figsize=np.array((7, 4)) * 0.9)

ax1.plot(ref_problem.pressure.pressure[1:-1, 1], '-k', label='reynolds, reference')

ax1.plot(problem.pressure.pressure[1:-1, 1], '--r', label='custom implementation')

ax1.set_xlabel("x")
ax1.set_ylabel("pressure")

ax1.legend()
ax1.set_xlabel("x")


