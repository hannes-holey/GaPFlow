#
# Copyright 2026 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from .md._lammps import lammps
import muGrid
import GaPFlow


def show_info():

    print(10 * "=")
    print('GaPFlow')
    print(10 * "=")

    print("Version:", GaPFlow.__version__)

    print()
    print(10 * "=")
    print('LAMMPS')
    print(10 * "=")

    lmp = lammps.lammps(name='mpi', cmdargs=['-log', 'none', "-screen", 'none'])
    print('Version:', lmp.version())
    print('Shared lib:', lmp.lib._name)
    print('MPI:', lmp.has_mpi_support)
    print('mpi4py:', lmp.has_mpi4py)
    print('Packages:', lmp.installed_packages)

    print()
    print(10 * "=")
    print('muGrid')
    print(10 * "=")

    print("Version:", muGrid.__version__)
    print('NetCDF4:', muGrid.has_netcdf)
    print('MPI:', muGrid.has_mpi)


def main():
    show_info()


if __name__ == "__main__":
    main()
