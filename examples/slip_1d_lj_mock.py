import io
import numpy as np
from scipy.special import erf

from GaPFlow.problem import Problem
from GaPFlow.io import read_yaml_input
from GaPFlow.db import Database
from GaPFlow.md import Mock


infile = """
options:
    output: data/slip_1d_lj
    write_freq: 100
    use_tstamp: False
grid:
    Lx: 1470.
    Ly: 1.
    Nx: 200
    Ny: 1
    xE: ['P', 'P', 'P']
    xW: ['P', 'P', 'P']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
geometry:
    type: inclined
    hmin: 12.
    hmax: 12.
    U: 0.12
    V: 0.
numerics:
    CFL: 0.5
    adaptive: 1
    tol: 1e-8
    dt: 0.1
    max_it: 5_000
properties:
    shear: 2.15
    bulk: 0.
    EOS: BWR
    T: 1.0
    rho0: 0.8
gp:
 press:
     fix_noise: True
     atol: 1.
     rtol: 0.
     obs_stddev: 2.e-2
     max_steps: 10
     active_dims: [0, ] # density
 shear:
     fix_noise: True
     atol: 1.
     rtol: 0.
     obs_stddev: 4.e-3
     max_steps: 10
     active_dims:     # optional, default is [0, 1, 3] (x) and [0, 2, 3] (y)
        x: [0, 1, 6]  # density, flux, slip length
db:
 init_size: 10
 init_method: lhc
 """

if __name__ == "__main__":

    with io.StringIO(infile) as ymlfile:
        input_dict = read_yaml_input(ymlfile)

    options = input_dict['options']
    grid = input_dict['grid']
    numerics = input_dict['numerics']
    prop = input_dict['properties']
    geo = input_dict['geometry']
    gp = input_dict['gp']
    db = input_dict['db']

    # Initialize extra field (smooth, periodic step function for slip length)
    nx = grid['Nx']
    ny = grid['Ny']
    a = 20.
    slip_length = np.zeros(nx)
    _erf = erf(np.linspace(-a, a, nx // 2))
    slip_length[:nx // 2] = _erf
    slip_length[nx // 2:] = -_erf
    slip_length = (1. + np.roll(slip_length, nx // 4)) / 2.

    # account for periodicity using ghost buffers
    extra = np.zeros((1, nx + 2, ny + 2))
    extra[0, 1:-1, :] = slip_length[:, None]
    extra[0, 0, :] = extra[0, -2, :]
    extra[0, -1, :] = extra[0, 1, :]

    md_runner = Mock(prop, geo, gp)

    database = Database(md_runner, db)

    problem = Problem(options,
                      grid,
                      numerics,
                      prop,
                      geo,
                      gp,
                      database,
                      extra_field=extra)
    # ... and run
    problem.run()
