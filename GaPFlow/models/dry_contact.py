#
# Copyright 2026 Christoph Huber
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

from datetime import datetime
from mpi4py import MPI
import numpy as np
import copy
import os

from ContactMechanics.Systems import NonSmoothContactSystem
from ContactMechanics import FreeFFTElasticHalfSpace
from SurfaceTopography import Topography

from ..topography import Topography as GaPFlowTopography


class DryContact:

    def __init__(self, input_dict, dir):

        self.domain_inlet = input_dict['force_balance']['init_dry_contact']['domain_inlet']
        self.domain_outlet = input_dict['force_balance']['init_dry_contact']['domain_outlet']
        self.domain_sides = input_dict['force_balance']['init_dry_contact']['domain_sides']

        assert self.domain_inlet > 0, "domain_inlet must be positive"
        assert self.domain_outlet > 0, "domain_outlet must be positive"
        assert self.domain_sides > 0, "domain_sides must be positive"

        self.geo = input_dict['geometry']
        self.grid = input_dict['grid']
        self.force = self.get_force(input_dict)  # check if force or pressure is specified

        self.elastic = input_dict['properties']['elastic']
        assert self.elastic['enabled'], "Elastic deformation must be enabled for dry contact initialization"

        self.dir = dir
        assert os.path.isdir(dir), f"Directory {dir} does not exist. Please check the path of the input YAML file."

        self.input_dict = input_dict

    def main(self):

        xx, yy = self.get_initial_grid()
        h = self.get_height(xx, yy)
        p, u = self.solve_contact(h)
        contact_bounds = self.get_bounding_box(p, xx, yy)
        domain_bounds = self.get_domain_bounds(contact_bounds)
        dict_new, h_new = self.update_input_dict(h, domain_bounds)

        debug_plot(p, u, contact_bounds, domain_bounds, self.grid, h, h_new)

        return dict_new

    def update_input_dict(self, h, domain_bounds):
        """Updates input dictionary (self.input_dict_new).
        Saves new height field to file and changes geometry type to 'from_file'.
        Updates grid size and spacing.

        Parameters
        ----------
        h : NDArray
            Original height field.
        domain_bounds : tuple
            New domain bounds.

        Returns
        -------
        input_dict_new : dict
            Updated input dictionary.
        """

        dict_new = copy.deepcopy(self.input_dict)
        grid = dict_new['grid']
        geo = dict_new['geometry']

        # Update grid size and spacing
        xmin, xmax, ymin, ymax = domain_bounds
        grid['Lx'] = xmax - xmin
        grid['Ly'] = ymax - ymin
        grid['dx'] = grid['Lx'] / grid['Nx']
        grid['dy'] = grid['Ly'] / grid['Ny']

        # Get new height field
        if self.geo['type'] == 'from_file':
            dx, dy = self.grid['dx'], self.grid['dy']
            ix_min = max(0, int(np.floor(xmin / dx)))
            ix_max = min(h.shape[0], int(np.ceil(xmax / dx)))
            iy_min = max(0, int(np.floor(ymin / dy)))
            iy_max = min(h.shape[1], int(np.ceil(ymax / dy)))
            h_new = h[ix_min:ix_max, iy_min:iy_max]

            grid['Nx'], grid['Ny'] = h_new.shape
            grid['dx'] = grid['Lx'] / grid['Nx']
            grid['dy'] = grid['Ly'] / grid['Ny']

        else:
            x = np.linspace(xmin + grid['dx'] / 2, xmax - grid['dx'] / 2, grid['Nx'])
            y = np.linspace(ymin + grid['dy'] / 2, ymax - grid['dy'] / 2, grid['Ny'])
            xx, yy = np.meshgrid(x, y, indexing='ij')
            # We need to use the original grid and geo here
            h_new, _, _ = GaPFlowTopography.compute_topography(xx, self.grid, self.geo, yy)

        # Save new height field to file and change input dict config
        folder = 'topography'
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"h_{timestamp_str}.npy"
        os.makedirs(os.path.join(self.dir, folder), exist_ok=True)
        np.save(os.path.join(self.dir, folder, filename), h_new)

        geo['type'] = 'from_file'
        geo['basepath'] = self.dir
        geo['filepath'] = os.path.join(folder, filename)

        return dict_new, h_new

    def get_domain_bounds(self, contact_bounds):

        xmin_, xmax_, ymin_, ymax_ = contact_bounds

        xspan, yspan = xmax_ - xmin_, ymax_ - ymin_
        xmid, ymid = (xmin_ + xmax_) / 2, (ymin_ + ymax_) / 2

        xmin = xmid - self.domain_inlet * xspan
        xmax = xmid + self.domain_outlet * xspan
        ymin = ymid - self.domain_sides * yspan
        ymax = ymid + self.domain_sides * yspan

        return xmin, xmax, ymin, ymax

    def get_bounding_box(self, p, xx, yy):
        """Returns bounding box of contact area based on pressure field.
        """
        contact_mask = p > 0
        assert np.any(contact_mask), "No contact detected. Please check force/pressure input and topography."
        assert p.shape == xx.shape == yy.shape, "Pressure field and grid must have the same shape."

        xmin = xx[contact_mask].min()
        xmax = xx[contact_mask].max()
        ymin = yy[contact_mask].min()
        ymax = yy[contact_mask].max()

        print(f"Contact area detected: {contact_mask.sum()} points")
        print(f"Contact area bounding box: x=[{xmin:.5f}, {xmax:.5f}], y=[{ymin:.5f}, {ymax:.5f}]")

        return xmin, xmax, ymin, ymax

    def solve_contact(self, h):
        Nx, Ny = self.grid['Nx'], self.grid['Ny']
        Lx, Ly = self.grid['Lx'], self.grid['Ly']

        substrate = FreeFFTElasticHalfSpace(
            (Nx, Ny),
            young=self.elastic['E'],
            physical_sizes=(Lx, Ly),
            # poisson=self.elastic['v']
        )

        # Invert height field for CM
        topography = Topography(-h, physical_sizes=(Lx, Ly))
        system = NonSmoothContactSystem(substrate, topography)
        print(self.force)
        result = system.minimize_proxy(external_force=self.force)

        print("Result success:", result.success)

        f, u = result.jac, result.x
        p = f / (self.grid['dx'] * self.grid['dy'])
        print(f"Max contact pressure: {p.max():.5f}")

        print("h min/max:", h.min(), h.max())
        print("Force:", self.force)
        print("Total force from solution:", f.sum())
        print("Mean pressure:", f.sum() / (Lx * Ly))
        print("Contact fraction:", (p > 0).mean())

        return p, u

    def get_force(self, input_dict):

        if 'force' in input_dict['force_balance']:
            force = float(input_dict['force_balance']['force'])
        elif 'pressure' in input_dict['force_balance']:
            pressure = float(input_dict['force_balance']['pressure'])
            area = self.grid['Lx'] * self.grid['Ly']
            force = pressure * area
        else:
            raise IOError("Need to specify either 'force' or 'pressure' in force_balance.")

        assert force > 0., "Negative force was determined. Please check force/pressure input."
        return force

    def get_initial_grid(self):
        """Returns xx and yy physical coordinates of inner, user-specified domain.
        """
        Nx, Ny = self.grid['Nx'], self.grid['Ny']
        x, y = np.arange(Nx), np.arange(Ny)
        xx_, yy_ = np.meshgrid(x, y, indexing='ij')

        dx, dy = self.grid['dx'], self.grid['dy']
        xx = xx_ * dx + dx / 2.0
        yy = yy_ * dy + dy / 2.0

        return xx, yy

    def get_height(self, xx, yy):
        """Returns height field of inner domain.
        """
        if self.geo['type'] == 'from_file':
            h = GaPFlowTopography.height_from_file(self.geo)
        else:
            h, _, _ = GaPFlowTopography.compute_topography(xx, self.grid, self.geo, yy)

        return h


def init_dry_contact(input_dict, dir):
    """Initializes dry contact problem by computing initial topography and force/pressure.

    Parameters
    ----------
    input_dict : dict
        Input dictionary containing problem configuration.
    dir : str
        Directory of the input YAML file, used for resolving relative paths.
    """
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        dry_contact = DryContact(input_dict, dir)
        input_dict_new = dry_contact.main()
    else:
        input_dict_new = None
    input_dict_new = comm.bcast(input_dict_new, root=0)
    return input_dict_new


def debug_plot(p, u, contact_bounds, domain_bounds, grid, h, h_new):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(p.T, origin='lower', extent=(0, grid['Lx'], 0, grid['Ly']))
    plt.colorbar(label='Contact Pressure')
    plt.title('Contact Pressure Distribution')

    plt.subplot(2, 2, 2)
    plt.imshow(u.T, origin='lower', extent=(0, grid['Lx'], 0, grid['Ly']))
    plt.colorbar(label='Displacement')
    plt.title('Displacement Distribution')

    # Contact bounds
    xmin, xmax, ymin, ymax = contact_bounds
    plt.subplot(2, 2, 1)
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r--', label='Contact Area')
    plt.legend()

    # Domain bounds
    xmin, xmax, ymin, ymax = domain_bounds
    plt.subplot(2, 2, 1)
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'g--', label='Domain Area')
    plt.legend()

    vmin, vmax = h.min(), h.max()

    # original h
    plt.subplot(2, 2, 3)
    plt.imshow(h.T, origin='lower', extent=(0, grid['Lx'], 0, grid['Ly']), cmap="terrain", vmin=vmin, vmax=vmax)
    plt.colorbar(label='Original Height Field')
    plt.title('Original Topography')

    # new h
    plt.subplot(2, 2, 4)
    plt.imshow(h_new.T, origin='lower', extent=(0, grid['Lx'], 0, grid['Ly']), cmap="terrain", vmin=vmin, vmax=vmax)
    plt.colorbar(label='Updated Height Field')
    plt.title('Updated Topography')

    plt.show()
