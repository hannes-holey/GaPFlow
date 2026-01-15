from GaPFlow.md._lammps import lammps

lmp = lammps.lammps(name='mpi', cmdargs=['-log', 'none', "-screen", 'none'])
print('LAMMPS Version: ', lmp.version())
print('OS:', lmp.get_os_info())
print('MPI: ', lmp.has_mpi_support)
print('mpi4py: ', lmp.has_mpi4py)
print('Installed packages:', lmp.installed_packages)
