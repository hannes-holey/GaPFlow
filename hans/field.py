#
# Copyright 2019-2021, 2023-2024 Hannes Holey
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


from mpi4py import MPI
import numpy as np


class Field:

    def __init__(self, disc, ndim):
        """Base class for all field objects.

        Parameters
        ----------
        disc : dict
            Discretization parameters.
        ndim : int
            Dimensionality.

        Attributes
        ----------
        comm : mpi4py.MPI.Cartcomm
            2D cartesian communicator.
        field : np.array
            Buffer holding the data of the class instance.
        disc : dict
            Discretization parameters.

        """

        self.disc = disc

        # global array size
        Nx = int(disc['Nx'])
        Ny = int(disc['Ny'])

        # read periodic boundary conditions
        periodicX = bool(disc['pX'])
        periodicY = bool(disc['pY'])

        # get cartesian communicator
        self.comm = self.get_2d_cart_comm(MPI.COMM_WORLD, (Nx, Ny),
                                          periods=(periodicX, periodicY))
        rank = self.comm.Get_rank()
        mpix, mpiy = self.comm.Get_coords(rank)
        size_x, size_y = self.comm.dims

        # local array sizes
        nx = Nx // size_x
        ny = Ny // size_y

        # fill right and topmost grid, s.t. local arrays sizes sum up to global size
        if mpix + 1 == size_x:
            nx += Nx % size_x
        if mpiy + 1 == size_y:
            ny += Ny % size_y

        # bounds of FD grid
        # left/bottom
        xleft = mpix * nx
        ybottom = mpiy * ny

        # right/top
        xright = xleft + nx - 1
        ytop = ybottom + ny - 1

        # adjust rightmost/topmost cell
        if mpix + 1 == size_x:
            xleft = Nx - nx
            xright = Nx - 1
        if mpiy + 1 == size_y:
            ybottom = Ny - ny
            ytop = Ny - 1

        ng = disc["nghost"]

        self._idx = np.linspace(xleft - ng, xright + ng, nx + 2 * ng, dtype=int)
        self._idy = np.linspace(ybottom - ng, ytop + ng, ny + 2 * ng, dtype=int)

        self.field = np.zeros(shape=(ndim, nx + 2 * ng, ny + 2 * ng), dtype=np.float64)

    def get_2d_cart_comm(self, comm, shape, periods=(True, True)):
        """
        Creates a two-dimensional cartesian MPI communicator.

        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator object
        periods : Iterable
            Directions with periodic boundary conditions (the default is (True, True)).

        Returns
        -------
        cartcomm : mpi4py.MPI.Cartcomm
            2D cartesian communicator
        rank : int
            rank of the MPI process
        size_x : int
            size of MPI grid in x direction
        size_y : int
            size of MPI grid in y direction

        """

        # get size of communicator
        size = comm.Get_size()

        Nx, Ny = shape

        # define optimal MPI grid from number of procs and global grid size
        possible_grids = [(i, size // i) for i in range(1, size + 1) if size % i == 0]
        grid_selector = [abs(i[0] / i[1] - Nx / Ny) for i in possible_grids]
        size_x, size_y = possible_grids[grid_selector.index(min(grid_selector))]

        # create cartesian communicator
        cartcomm = comm.Create_cart(dims=(size_x, size_y), periods=periods)

        return cartcomm

    def get_neighbors(self):
        """
        Get MPI ranks of neighboring cells.

        Returns
        -------
        tuple
            Contains left, right, bottom and top neighbors, src and destination respectively.

        """
        # get MPI ranks of neighboring cells
        left = self.comm.Shift(0, -1)
        right = self.comm.Shift(0, 1)
        bottom = self.comm.Shift(1, -1)
        top = self.comm.Shift(1, 1)

        return (left, right, bottom, top)

    @property
    def inner(self):
        ng = self.disc["nghost"]
        return np.ascontiguousarray(self.field[:, ng:-ng, ng:-ng])

    @inner.setter
    def inner(self, field):
        ng = self.disc["nghost"]
        self.field[:, ng:-ng, ng:-ng] = field

    @property
    def edgeE(self):
        return (self.field + np.roll(self.field, -1, axis=1)) / 2.

    @property
    def edgeN(self):
        return (self.field + np.roll(self.field, -1, axis=2)) / 2.

    @property
    def edgeW(self):
        return (self.field + np.roll(self.field, 1, axis=1)) / 2.

    @property
    def edgeS(self):
        return (self.field + np.roll(self.field, 1, axis=2)) / 2.

    @property
    def verticeNE(self):
        return (self.field +
                np.roll(self.field, -1, axis=1) +
                np.roll(self.field, -1, axis=2) +
                np.roll(self.field, (-1, -1), axis=(1, 2))) / 4.

    @property
    def verticeSW(self):
        return (self.field +
                np.roll(self.field, 1, axis=1) +
                np.roll(self.field, 1, axis=2) +
                np.roll(self.field, (1, 1), axis=(1, 2))) / 4.

    @property
    def id_grid(self):
        return np.meshgrid(self._idx, self._idy, indexing='ij')

    @property
    def without_ghost(self):
        ng = self.disc["nghost"]
        wo_ghost_x = slice(self._idx[ng], self._idx[-ng])
        wo_ghost_y = slice(self._idy[ng], self._idy[-ng])
        return wo_ghost_x, wo_ghost_y

    @property
    def centerline_x(self):
        ng = self.disc["nghost"]
        return self.field[:, ng:-ng, self.disc["Ny"] // 2 + ng]

    @property
    def centerline_y(self):
        ng = self.disc["nghost"]
        return self.field[:, self.disc["Nx"] // 2 + ng, ng:-ng]


class ScalarField(Field):
    """Scalar field derived from Field base class."""

    def __init__(self, disc):
        """Constructor.

        Parameters
        ----------
        disc : dict
            Description of parameter `disc`.

        Returns
        -------
        type
            Description of returned object.

        """
        super().__init__(disc, 1)


class VectorField(Field):
    """Vector field derived from Field base class."""

    def __init__(self, disc):
        """Constructor.

        Parameters
        ----------
        disc : dict
            Discretization parameters.

        """
        super().__init__(disc, 3)


class TensorField(Field):
    """Symmetric tensor field derived from Field base class."""

    def __init__(self, disc):
        """Constructor.

        Parameters
        ----------
        disc : dict
            Discretization parameters.

        """
        super().__init__(disc, 6)


class DoubleTensorField(Field):
    """Symmetric tensor field derived from Field base class."""

    def __init__(self, disc):
        """Constructor.

        Parameters
        ----------
        disc : dict
            Discretization parameters.

        """
        super().__init__(disc, 12)
