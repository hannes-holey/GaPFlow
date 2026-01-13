from .md._lammps import lammps
import muGrid
import GaPFlow


def show_info():

    print(10 * "=")
    print('GaPFlow')
    print(10 * "=")

    print("Version: ", GaPFlow.__version__)

    print(10 * "=")
    print('LAMMPS')
    print(10 * "=")

    lmp = lammps.lammps(name='mpi', cmdargs=['-log', 'none', "-screen", 'none'])

    print('OS:', lmp.get_os_info())
    print('Shared lib: ', lmp.lib._name)
    print('Version: ', lmp.version())
    print('MPI: ', lmp.has_mpi_support)
    print('mpi4py: ', lmp.has_mpi4py)
    print('Packages: ', lmp.installed_packages)

    print(10 * "=")
    print('muGrid')
    print(10 * "=")

    print("Version: ", muGrid.__version__)
    print('MPI: ', muGrid.has_mpi)
    # print('NetCDF4: ' muGrid.has_netcdf) # >= 0.97.0
    # print('GPU: ', muGrid.has_gpu)


def main():
    show_info()


if __name__ == "__main__":
    main()
