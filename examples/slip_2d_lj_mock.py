import io
import numpy as np

from GaPFlow.problem import Problem
from GaPFlow.io import read_yaml_input
from GaPFlow.db import Database
from GaPFlow.md import Mock


infile = """
options:
    output: data/slip_2d_lj
    write_freq: 100
    use_tstamp: False
grid:
    Lx: 1470.
    Ly: 1470.
    Nx: 128
    Ny: 128
    xE: ['P', 'P', 'P']
    xW: ['P', 'P', 'P']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
geometry:
    type: inclined
    hmin: 10.
    hmax: 10.
    U: 0.12
    V: 0.
numerics:
    CFL: 0.5
    adaptive: 1
    tol: 1e-8
    dt: 0.05
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
     max_steps: 5
     active_dims: [0, ] # only density
 shear:
     fix_noise: True
     atol: 1.
     rtol: 0.
     obs_stddev: 4.e-3
     max_steps: 5
     active_dims:     # optional, default is [0, 1, 3] (x) and [0, 2, 3] (y)
        x: [0, 1, 6]  # density, flux_x, slip length
        y: [0, 2, 6]  # density, flux_y, slip length
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

    # Create checkerboard pattern
    nx = grid['Nx']
    ny = grid['Ny']
    slip_length = np.zeros((nx, ny))
    slip_length[:nx // 2, :ny // 2] = 1.
    slip_length[nx // 2:, ny // 2:] = 1.

    _x = np.linspace(-grid['Lx'] / 2., grid['Lx'] / 2., nx)
    _y = np.linspace(-grid['Ly'] / 2., grid['Ly'] / 2., ny)
    xx, yy = np.meshgrid(_x, _y)
    x = np.dstack([xx, yy]).T

    # Smoothing with Gaussian filter
    s = 20.
    inv = np.eye(2) / s**2
    gauss = 1. / (2 * np.pi * s**2) * np.exp(-0.5 * np.einsum('j..., j...',
                                                              x,
                                                              np.einsum('ij, j... -> i...', inv, x)))
    gauss /= np.sum(gauss)
    smooth_slip_length = np.fft.ifft2(np.fft.fft2(gauss) * np.fft.fft2(slip_length)).real

    # fill extra field with ghost buffers
    extra = np.zeros((1, nx + 2, ny + 2))
    extra[0, 1:-1, 1:-1] = np.roll(smooth_slip_length, ny // 4, axis=1)
    extra[0, 0, :] = extra[0, -2, :]
    extra[0, -1, :] = extra[0, 1, :]
    extra[0, :, 0] = extra[0, :, -2]
    extra[0, :, -1] = extra[0, :, 1]

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
