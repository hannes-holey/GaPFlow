

# Requires:  uv pip install git+ssh://git@github.com/threich/reynolds.git@miniter for the reynolds reference solver

import io
import pytest
import numpy as np

from GaPFlow.models.continuum_wall_stress import NewtonParabolicFlowFluidGap, SlipLengthSlip, Operations, EyringSlip


from GaPFlow.io import read_yaml_input
from GaPFlow import Problem
from GaPFlow.config_utils import build_config
import matplotlib.pyplot as plt

from GaPFlow.viz import evaluate_quadfield_on_nodes

from reynolds.solver import SteadyReynolds1D

import tol_colors as tc

tol= 1e-13

config = f"""
options:
    output: data/journal
    write_freq: 1000
    silent: False
grid:
    Lx: 1.
    Ly: 100.
    Nx: 1
    Ny: 32
    xE: ['P', 'P', 'P',]
    xW: ['P', 'P', 'P',]
    yS: ['P', 'P', 'P',]
    yN: ['P', 'P', 'P',]
geometry:
    type: journal
    hmin: 1.
    hmax: 2.
    U_top: 0.
    V_top: 1.
    U_bot: 0.
    V_bot: 0.
numerics:
    solver: implicit
    dt: 1.0
    tol: 1e-6
    max_it: 10000

properties:
    EOS: DH
    rho0: 1.
    P0: 1. # the pressure unit is viscosity * velocity / hmin # Here we set this value to 1. as anb example, but we will update later to give physically more meaningful values.
    C1: 345423.1433506045  # Which is the classical valye 3.5e10 / the classical P0 value of 101325
    C2: 1.23
    shear: 1. # This together with the velocity determines the pressure unit
    bulk: 0.0

implicit_solver:
    linear_solver: direct
    scaling: False
    newton_relax: 0.5
    p_init: 1.
    tol_abs_resnorm: {tol}
    equations:
        cavitation: False
    max_iter: 100
"""


def make_reynolds(gpf_problem):

    geo = gpf_problem.geo
    grid = gpf_problem.grid
    prop = gpf_problem.prop

    slip_law = gpf_problem.custom_models['slip_top']

    hmin = geo['hmin']
    hmax = geo['hmax']

    c = (hmin + hmax) / 2
    rb = 1 / (2  * np.pi) * grid['Ly']
    rj = rb - c

    e = (hmax - hmin ) /2

    # Eccentricity
    eps =  e /  c

    params = {
        'geometry': {
            'type': 'journal',
            # 'Lx': grid['Lx'],
            'rb': rb,
            'rj': rj,
            'eps': eps,
        },
        'BC': {
            'u1': geo['V_bot'],
            'u2': geo['V_top'],
            'p': {
                'W': ['D', 1],
                'E': ['D', 1],
            },
        },
        # --- equation of state (Murnaghan-Tait) ---
        'density': {
            'type': 'DH',
            'rho0': prop['rho_ref'],
            'p0': prop['P0'],
            'C1': prop['C1'],
            'C2': prop['C2'],
        },
        'numerics': {
            'Nx': grid['Ny'] + 2,
            'max_iter': 100000,
            'tol': tol,
            'verbose': True,
            'every': 10,
            'nondim': False,
            'alpha_visc': 1.,
            'alpha_press': .1,
            'alpha_slip': 0.001,
        },


    }

    visc = {
        'piezo':'None',
        'thinning': 'None',
        'mu0':    prop['shear']}

    slip = {
        'type': 'Eyring',
        'At': 0,
        'Bt': slip_law.tc,
        'Av': slip_law.vc,
        'Bv': 0,
        'Cv': 0,
        'flatonly': False,
    }

    params['viscosity'] = visc
    params['slip'] = slip

    print('RUNNING REYNOLDS')
    pb = SteadyReynolds1D(params)
    pb.run(miniter=22)
    return pb


# %%
pbs = {}
for props in [
    # (2000., 2.),
    (2., 2.),
              # (1.,10.)
              ]:

    tauE, vE = props
    gap = NewtonParabolicFlowFluidGap(mu0=1.)
    operations = Operations()
    slip_top = EyringSlip(tc=tauE, vc=vE)
    slip_bot =  EyringSlip(tc=tauE, vc=vE)

    config_hans_slip = build_config(config, **{'implicit_solver.physics.slip': True,
                                               'options.output': 'data/journal_hans_slip'})

    with io.StringIO(config_hans_slip) as ymlfile:
        inp = read_yaml_input(ymlfile)

    problem = Problem(options=inp['options'],
                       grid=inp['grid'],
                       numerics=inp['numerics'],
                       prop=inp['properties'],
                       geo=inp['geometry'],
                       fem_solver=inp['fem_solver'],
                       energy_spec=inp['energy_spec'],
                       custom_models=dict(wall_stress=gap,
                                          operations=operations,
                                          slip_top=slip_top,
                                          slip_bot=slip_bot,
                                          ),
                      )
    geo=problem.geo
    grid = problem.grid
    amp = (geo['hmax'] - geo['hmin']) / 2.
    shift = (geo['hmax'] + geo['hmin']) / 2.
    freq = 2. * np.pi / grid['Ly']

    h = shift + amp * np.cos(freq * problem.topo.yy)
    problem.topo.set_mapped_height(h)
    problem.run()
    pb_reynolds = make_reynolds(problem)
    pbs[props] = (problem, pb_reynolds)

# %%
fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=np.array((10, 4)) * 0.9)
colors = iter(tc.bright)
for props, problems in pbs.items():
    c = next(colors)
    ax= axes[0, 0]

    thing = 'p'
    ax.plot(evaluate_quadfield_on_nodes(problems[0], thing).flat, '-', label=f"{props}, {thing}", c=c)
    # ax.set_ylabel(things[0])
    ax.plot(problems[1].pressure[1:-1], '+', c=c)


    # Shear stresses:

    thing = 'tau_yz_bot'
    axes[0, 1].plot(evaluate_quadfield_on_nodes(problems[0], thing).flat, '-', label=f"{props}, {thing}", c=c)
    thing = 'tau_yz_top'
    axes[0, 2].plot(evaluate_quadfield_on_nodes(problems[0], thing).flat, '-', label=f"{props}, {thing}", c=c)

    tau_bot, tau_top = problems[1].shear_stress_wall
    axes[0, 1].plot(tau_bot[1:-1], '+', c=c)
    axes[0, 2].plot(tau_top[1:-1], '+', c=c)

    # Velocities at the wall

    thing = 'Vf_bot'
    axes[1, 1].plot(evaluate_quadfield_on_nodes(problems[0], thing).flat , '-', label=f"{props}, {thing}", c=c)
    thing = 'Vf_top'
    axes[1, 2].plot(evaluate_quadfield_on_nodes(problems[0], thing).flat, '-', label=f"{props}, {thing}", c=c)

    uf1 = problems[1].u_lower
    uf2 = problems[1].u_upper

    axes[1, 1].plot(uf1[1:-1], '+', c=c)
    axes[1, 2].plot(uf2[1:-1], '+', c=c)


axes[0,0].set_ylabel('p')

axes[0, 1].set_ylabel('tau bot')
axes[0,2].set_ylabel('tau top')

axes[1, 1].set_ylabel('V bot')
axes[1,2].set_ylabel('V top')
# C

axes[0,0].legend(title='tc, vc')
fig.show()
