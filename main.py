import matplotlib.pyplot as plt
from muGrid import GlobalFieldCollection, FileIONetCDF, OpenMode

from hans_mugrid.gap import GapHeight
from hans_mugrid.density import Solution


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
            'Nx': 100,
            'Ny': 1,
            'CR': 1.e-2,
            'eps': 0.7
            }

    disc['dx'] = disc['Lx'] / disc['Nx']
    disc['dy'] = disc['Ly'] / disc['Ny']
    disc['dt'] = 3e-10

    # Two dimensional grid
    nb_grid_pts = (disc['Nx'], disc['Ny'])
    fc = GlobalFieldCollection(nb_grid_pts)

    # intialize all fields
    geometry = GapHeight(fc, disc)
    solution = Solution(fc, disc, prop)

    solution.initialize(prop['rho0'], 0.1, 0.)

    file = FileIONetCDF('example.nc', OpenMode.Overwrite)
    file.register_field_collection(fc, field_names=['solution', 'pressure'])

    Nt = 2000

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(Nt):
        solution.update(0)

        if i % 100 == 0:

            print(f"{i:>5d} {solution.residual:.3e}")

            color_q = plt.cm.Blues(i / Nt)
            ax[0, 0].plot(solution.density, color=color_q)
            ax[0, 1].plot(solution.flux_x, color=color_q)
            ax[0, 2].plot(solution.flux_y, color=color_q)

            color_p = plt.cm.Greens(i / Nt)
            color_t = plt.cm.Oranges(i / Nt)
            ax[1, 0].plot(solution.pressure.pressure, color=color_p)
            ax[1, 1].plot(solution.wall_stress.lower[4, :, disc['Ny'] // 2], color=color_t)
            ax[1, 2].plot(solution.wall_stress.upper[4, :, disc['Ny'] // 2], color=color_t)

            file.append_frame().write()

    plt.show()
