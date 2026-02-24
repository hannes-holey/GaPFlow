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
from .utils import read_output_files
from .moltemplate import write_template, build_template
from .base import MolecularDynamics

import numpy as np
import os
import shutil
import scipy.constants as sci
from copy import deepcopy


class LennardJones(MolecularDynamics):
    """Run MD simulations with LAMMPS for a pure LJ system."""

    name = 'lj'

    def __init__(self, params):
        """Constructor.

        Parameters
        ----------
        params : dict
            Parameters to control the setup of the MD simulations (read from YAML input).
        """
        self.is_mock = False
        self.main_file = 'in.run'
        self.num_worker = params['ncpu']
        self.params = params

    def build_input_files(self, dataset, location, X):
        # write variables file
        variables_str = f"""
variable\tinput_gap equal {X[3]}
variable\tinput_dens equal {X[0]}
variable\tinput_fluxX equal {X[1]}
variable\tinput_fluxY equal {X[2]}
"""
        excluded = ['infile', 'wallfile', 'ncpu', 'system']

        # equal-style variables
        for k, v in self.params.items():
            if k not in excluded:
                variables_str += f'variable\t{k} equal {v}\n'

        variables_str += 'variable\tslabfile index in.wall\n'

        with open(os.path.join(location, 'data', 'in.param'), 'w') as f:
            f.writelines(variables_str)

        # Move inputfiles to proto dataset
        dataset.put_item(self.params['wallfile'], 'in.wall')
        dataset.put_item(self.params['infile'], 'in.run')

    def read_output(self):
        return read_output_files()


class GoldAlkane(MolecularDynamics):
    """Run MD simulations with LAMMPS for n-alkanes confined between gold surfaces.

    Input files are build with the help of ASE and moltemplate.
    """

    name = 'mol'

    def __init__(self, params):
        """Constructor.

        Parameters
        ----------
        params : dict
            Parameters to control the setup of the MD simulations (read from YAML input).
        """
        self.is_mock = False
        self.main_file = 'run.in.all'
        self.params = params
        self.num_worker = params['ncpu']

    def build_input_files(self, dataset, location, X):
        proto_ds_datapath = os.path.join(location, 'data')

        # Move inputfiles to proto dataset
        os.makedirs(os.path.join(proto_ds_datapath,
                                 'moltemplate_files'))

        os.makedirs(os.path.join(proto_ds_datapath,
                                 'static'))

        dataset.put_item(self.params['fftemplate'],
                         os.path.join("moltemplate_files",
                                      os.path.basename(self.params['fftemplate'])
                                      )
                         )

        dataset.put_item(self.params['topo'],
                         os.path.join("moltemplate_files",
                                      os.path.basename(self.params['topo'])
                                      )
                         )

        for f in os.listdir(self.params["staticFiles"]):
            dataset.put_item(os.path.join(self.params["staticFiles"], f),
                             os.path.join("static", f))

        args = deepcopy(self.params)
        args["density"] = float(X[0])
        args["fluxX"] = float(X[1])
        args["fluxY"] = float(X[2])
        args["gap_height"] = float(X[3])

        if self.params['wall_rotation']:
            dh_dx = float(X[4])
            args["rotation"] = -np.arctan(dh_dx) / np.pi * 180.

        cwd = os.getcwd()
        os.chdir(proto_ds_datapath)
        self.num_worker = write_template(args)
        build_template(args)
        shutil.rmtree('output_ttree')
        os.chdir(cwd)

    def read_output(self):
        sf = sci.calorie * 1e-4  # from kcal/mol/A^3 to g/mol/A/fs^2
        return read_output_files(sf=sf)
