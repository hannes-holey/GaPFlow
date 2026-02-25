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
from .base import MolecularDynamics
from .utils import read_output_files

import os


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
