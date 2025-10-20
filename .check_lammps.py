try:
    from lammps import lammps
except ImportError:
    lammps = None

if lammps is not None:
    lmp = lammps(cmdargs=['-log', 'none', "-screen", 'none'])
    print('LAMMPS Version: ', lmp.version())
    print('OS:', lmp.get_os_info())
    print('MPI: ', lmp.has_mpi_support)
    print('mpi4py: ', lmp.has_mpi4py)
    print('Installed packages:', lmp.installed_packages)
else:
    print('LAMMPS Python module is not installed')
