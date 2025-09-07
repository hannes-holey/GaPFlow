from datetime import datetime
from muGrid import GlobalFieldCollection, FileIONetCDF, OpenMode
from hans_mugrid.gap import Gap
from hans_mugrid.density import Solution
from hans_mugrid.db import Database
from hans_mugrid.plotting import plot_single_frame, plot_evolution, plot_history, animate


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

    # Init
    # Two dimensional grid
    nb_ghost = 1  # ghost buffer
    nb_grid_pts = (disc['Nx'] + 2 * nb_ghost,
                   disc['Ny'] + 2 * nb_ghost)

    tic = datetime.now()
    fc = GlobalFieldCollection(nb_grid_pts)

    # intialize all fields
    geometry = Gap(fc, disc)
    database = Database(minimum_size=5)
    solution = Solution(fc, disc, prop, bc, data=database)

    solution.write()

    # Run
    while not solution.converged and solution.step < 200:
        solution.update()

        if solution.step % 1 == 0:
            solution.write()

    toc = datetime.now()

    solution.history_to_csv('example.csv')

    walltime = toc - tic
    speed = solution.step / walltime.total_seconds()

    print(33 * '=')
    print(f"Total walltime     : ", str(walltime).split('.')[0])
    print(f"({speed:.2f} steps/s)")
    print(f" - GP train (press): ", str(solution.pressure.cumtime_train).split('.')[0])
    print(f" - GP infer (press): ", str(solution.pressure.cumtime_infer).split('.')[0])
    print(f" - GP train (shear): ", str(solution.wall_stress.cumtime_train).split('.')[0])
    print(f" - GP infer (shear): ", str(solution.wall_stress.cumtime_infer).split('.')[0])
    print(33 * '=')
