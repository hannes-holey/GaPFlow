
from hans_mugrid.problem import Problem


if __name__ == "__main__":

    # Input:
    # - options: name, output, outfreq
    # - grid: Lx, Nx, dx, ...
    # - boundary_conditions: h, Dh, (E,W,N,S), values, periodic
    # - properties:
    #   - fluid: ...
    #   - solid: (later)
    # - numerics: dt, CFL, adaptive
    # - gp:
    #   - press: tol, noise, freq, wait, max_steps
    #   - shear: ...
    # - db: dtool, location, template, remote, Ninit, QMC sampling
    # - md: ncpu, setup (lammps/moltemplate), temperature, velocity, sampling_time, dump_freq

    prop = {'shear': 0.0794,
            'bulk': 0.,
            'P0': 101325,
            'rho0': 877.7007,
            'T0': 323.15,
            'C1': 3.5e10,
            'C2': 1.23}

    disc = {'Lx': 1.e-3,
            'Ly': 1.,
            'Nx': 100,
            'Ny': 1,
            'CR': 1.e-2,
            'eps': 0.7
            }

    bc = {'x0': ['D', 'N', 'N'],
          'x1': ['D', 'N', 'N'],
          'y0': ['P', 'P', 'P'],
          'y1': ['P', 'P', 'P'],
          'rhox0': prop['rho0'],
          'rhox1': prop['rho0'],
          'rhoy0': prop['rho0'],
          'rhoy1': prop['rho0']}

    disc['dx'] = disc['Lx'] / disc['Nx']
    disc['dy'] = disc['Ly'] / disc['Ny']
    disc['dt'] = 3e-10

    problem = Problem(disc, prop, bc)

    # maybe pass some args to run method as they are not problem specific
    # e.g., max_steps, tolerance, adaptive, integrator (only MC a.t.m.)
    problem.run()
