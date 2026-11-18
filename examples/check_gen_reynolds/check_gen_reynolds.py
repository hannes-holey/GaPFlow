
import io
import pytest
import numpy as np

from GaPFlow.io import read_yaml_input
from GaPFlow import Problem
from GaPFlow.config_utils import build_config
import matplotlib.pyplot as plt
import tol_colors as tc

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
        cavitation: True
        generalized_reynolds: True
        reynolds: True
    max_iter: 100

"""

# %% Checking expected values:
overrides = {
    'options.silent': False,
    'options.print_config': False,
    'properties.thinning.name': 'Eyring',
}
label = 'test'
tauE = 10000

config = build_config(config, **overrides,
                      **{'implicit_solver.equations.generalized_reynolds': True,
                         'properties.thinning.tauE': tauE,
                         'options.output': f'data/{label}',
                         })
problem = Problem.from_string(config)
problem.run()

# %%
solver = problem.solver
qf = solver.quad_mgr.collect_quad_fields()

mu0 = problem.prop['shear']

np.testing.assert_allclose(qf['rho_eff'], qf['rho'])
np.testing.assert_allclose(qf['rho_star_y'], qf['rho'])
np.testing.assert_allclose(qf['rho_star_x'], qf['rho'])

np.testing.assert_allclose(qf['rho_o_eta_eff_x'], qf['rho'] / mu0, rtol=1e-2)
np.testing.assert_allclose(qf['rho_o_eta_eff_y'], qf['rho']/ mu0)


# %% Comparison with reynolds

overrides = {
    'options.silent': False,
    'options.print_config': False,
    'properties.thinning.name': 'Eyring',
}

tauEs = [1000, 10]
results = {}
for label, generalized, custom in [('Reynolds', False, False),
                                   #('generalized, default', True, False),
                                   ('generalized, custom', True, False)
                                   ]:

    for tauE in tauEs:
        config = build_config(config, **overrides,
                              **{'implicit_solver.equations.generalized_reynolds': generalized,
                                 'properties.thinning.tauE': tauE,
                                 'options.output': f'data/{label}',
                                 })
        problem = Problem.from_string(config)
        problem.run()
        results[label+ f", tauE={tauE}"] = problem

def color_from_tauE(tauE):
    vmin = np.min(tauEs)
    vmax = np.max(tauEs)
    val = (tauE - vmin) / (vmax - vmin)
    return tc.BuRd(val)



fig, ax1 = plt.subplots(1, 1 , sharex=True, sharey=False, figsize=np.array((7, 4)) * 0.9)

for label, pb in results.items():
    style = dict(label=f"{label}")

    if "generalized, default" in label:
        style |= dict(ls ="--")
    elif "generalized, custom" in label:
        style |= dict(ls ="none", marker = "+")
    else:
        style |= dict(ls='-')

    tauE = float(label.split("=")[-1])
    style |= dict(c=color_from_tauE(tauE))

    ax1.plot(pb.pressure.pressure[1:-1, 1], **style)


ax1.set_xlabel("x")
ax1.set_ylabel("pressure")

ax1.legend()
ax1.set_xlabel("x")

# %%

fig.show()
