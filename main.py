import matplotlib.pyplot as plt
from muGrid import GlobalFieldCollection, FileIONetCDF, OpenMode
import netCDF4
import numpy as np

from hans_mugrid.gap import GapHeight
from hans_mugrid.density import Solution


def plot(filename='example.nc'):

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    tau_nc = np.asarray(data.variables['wall_stress'])

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    Nt, _, _ = p_nc.shape
    for i in range(Nt):
        color_q = plt.cm.Blues(i / Nt)
        ax[0, 0].plot(q_nc[i, 0, 0, 1:-1, disc['Ny'] // 2], color=color_q)
        ax[0, 1].plot(q_nc[i, 1, 0, 1:-1, disc['Ny'] // 2], color=color_q)
        ax[0, 2].plot(q_nc[i, 2, 0, 1:-1, disc['Ny'] // 2], color=color_q)

        color_p = plt.cm.Greens(i / Nt)
        color_t = plt.cm.Oranges(i / Nt)

        ax[1, 0].plot(p_nc[i, 1:-1, disc['Ny'] // 2], color=color_p)
        ax[1, 1].plot(tau_nc[i, 4, 0, 1:-1, disc['Ny'] // 2], color=color_t)
        ax[1, 2].plot(tau_nc[i, 10, 0, 1:-1, disc['Ny'] // 2], color=color_t)

    plt.show()


if __name__ == "__main__":

    prop = {'shear': 0.0794,
            'bulk': 0.,
            'P0': 101325,
            'rho0': 877.7007,
            'T0': 323.15,
            'C1': 3.5e10,
            'C2': 1.23}

    disc = {'Lx': 1.e-3,
            'Ly': 1.,
            'Nx': 50,
            'Ny': 20,
            'CR': 1.e-2,
            'eps': 0.7
            }

    bc = {'x0': ['D', 'N', 'N'],
          'x1': ['D', 'N', 'N'],
          'y0': ['D', 'N', 'N'],
          'y1': ['D', 'N', 'N'],
          'rhox0': prop['rho0'],
          'rhox1': prop['rho0'],
          'rhoy0': prop['rho0'],
          'rhoy1': prop['rho0']}

    disc['dx'] = disc['Lx'] / disc['Nx']
    disc['dy'] = disc['Ly'] / disc['Ny']
    disc['dt'] = 3e-10

    # Two dimensional grid
    nb_ghost = 1  # ghost buffer
    nb_grid_pts = (disc['Nx'] + 2 * nb_ghost,
                   disc['Ny'] + 2 * nb_ghost)

    fc = GlobalFieldCollection(nb_grid_pts)

    # intialize all fields
    geometry = GapHeight(fc, disc)
    solution = Solution(fc, disc, prop, bc)

    solution.initialize(prop['rho0'], 0.1, 0.)

    file = FileIONetCDF('example.nc', OpenMode.Overwrite)
    file.register_field_collection(fc, field_names=['solution', 'pressure', 'wall_stress'])

    print(f"{solution.step:>5d} {solution.residual:.3e}")
    file.append_frame().write()

    while not solution.converged and solution.step < 5000:
        solution.update()

        if solution.step % 100 == 0:
            print(f"{solution.step:>5d} {solution.residual:.3e}")
            file.append_frame().write()
