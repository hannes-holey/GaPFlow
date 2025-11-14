#
# Copyright 2025 Hannes Holey
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
import os
import sys
import abc
import shutil
import warnings
import dtoolcore
from mpi4py import MPI
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from ruamel.yaml import YAML
from getpass import getuser
from socket import gethostname
import scipy.constants as sci
from copy import deepcopy

try:
    from lammps import lammps
except ImportError:
    warnings.warn('Failed to import lammps. Only Mock MD object available.')

import jax.random as jr
import jax.numpy as jnp
import numpy as np

from ..models.pressure import eos_pressure
from ..models.viscous import stress_bottom, stress_top
from .moltemplate import write_template, build_template
from .utils import read_output_files
from ..utils import bordered_text, make_dumpable

yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=4, sequence=4, offset=2)


def main():

    comm = MPI.Comm.Get_parent()

    run_serial(sys.argv[1])

    comm.Barrier()
    comm.Free()


def run_parallel(fname, nworker):

    worker_file = os.path.abspath(__file__)

    sub_comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[worker_file, fname],
                                   maxprocs=nworker)

    # Wait for MD to complete and free spawned communicator
    sub_comm.Barrier()
    sub_comm.Free()


def run_serial(fname):

    nargs = ["-log", "log.lammps"]
    lmp = lammps(name='mpi', cmdargs=nargs)
    assert lmp.has_package('EXTRA-FIX'), "Lammps needs to be compiled with package 'EXTRA-FIX'"

    lmp.file(fname)


if __name__ == "__main__":
    # main is called by an individual spawned process for parallel MD runs
    main()


class MolecularDynamics:
    __metaclass__ = abc.ABCMeta

    name = str
    params: dict
    main_file: str
    num_worker: int
    path: str
    is_mock: bool
    _dtool_basepath: str = '/tmp/'
    readme_template: str = ""
    input_names: list[str] = ['ρ', 'jx', 'jy', 'h', '∂h/∂x', '∂h/∂y'] + [f'extra_{i}' for i in range(10)]
    ascii_art: str = r"""
  _        _    __  __ __  __ ____  ____
 | |      / \  |  \/  |  \/  |  _ \/ ___|
 | |     / _ \ | |\/| | |\/| | |_) \___ \
 | |___ / ___ \| |  | | |  | |  __/ ___) |
 |_____/_/   \_\_|  |_|_|  |_|_|   |____/

"""

    @property
    def dtool_basepath(self):
        return self._dtool_basepath

    @dtool_basepath.setter
    def dtool_basepath(self, name):
        self._dtool_basepath = name

    @abc.abstractmethod
    def build_input_files(self, dataset, location, X):
        raise NotImplementedError

    @abc.abstractmethod
    def read_output(self):
        raise NotImplementedError

    def add_metadata_to_readme(self):
        raise NotImplementedError

    def pretty_print(self, proto_datapath, X):
        text = ['Run next MD simulation in:', f'{proto_datapath}']
        text.append(self.ascii_art)
        text.append('---')
        for i, (Xi, name) in enumerate(zip(X, self.input_names)):
            text.append(f'Input {i + 1}: {Xi:+.3e}    ({name})')
        print(bordered_text('\n'.join(text)))

    def write_dtool_readme(self, dataset_path, Xnew, Ynew, Yerrnew):
        if len(self.readme_template) == 0:
            metadata = {}
        else:
            metadata = yaml.load(self.readme_template)

        # Update metadata
        metadata["owners"] = [{'username': getuser()}]
        metadata["creation_date"] = date.today()
        metadata["expiration_date"] = metadata["creation_date"] + relativedelta(years=10)

        out_fname = os.path.join(dataset_path, 'README.yml')

        metadata.update({'parameters': make_dumpable(self.params)})

        metadata['X'] = make_dumpable(Xnew)
        metadata['Y'] = make_dumpable(Ynew)
        metadata['Yerr'] = make_dumpable(Yerrnew)

        with open(out_fname, 'w') as outfile:
            yaml.dump(metadata, outfile)

    def create_dtool_dataset(self, tag):
        ds_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self.name}-{tag:03}'

        proto_ds = dtoolcore.create_proto_dataset(name=ds_name,
                                                  base_uri=self.dtool_basepath)

        proto_ds_path = proto_ds.uri.removeprefix('file://' + gethostname())

        return proto_ds, proto_ds_path

    def run(self, X, tag):

        # Setup MD simulation
        dataset, location = self.create_dtool_dataset(tag)
        self.build_input_files(dataset, location, X)

        self.pretty_print(location, X)

        # Move to dtool datapath...
        basedir = os.getcwd()
        os.chdir(os.path.join(location, 'data'))

        # ...Run MD...
        if self.num_worker > 1:
            run_parallel(self.main_file, self.num_worker)
        elif self.num_worker == 1:
            run_serial(self.main_file)
        else:
            pass

        # ...Read output / post-process MD result...
        Y, Ye = self.read_output()

        # ...and return to cwd
        os.chdir(basedir)

        # Finalize dataset
        self.write_dtool_readme(location, X, Y, Ye)
        dataset.freeze()

        return Y, Ye


class Mock(MolecularDynamics):

    name = 'mock'

    ascii_art: str = r"""
  __  __  ___   ____ _  __
 |  \/  |/ _ \ / ___| |/ /
 | |\/| | | | | |   | ' /
 | |  | | |_| | |___| . \
 |_|  |_|\___/ \____|_|\_\

"""

    def __init__(self, prop, geo, gp):

        self.is_mock = True

        self.noise = (gp['press']['obs_stddev'] if gp['press_gp'] else 0.,
                      gp['shear']['obs_stddev'] if gp['shear_gp'] else 0.)

        self.num_worker = 0
        self.geo = geo
        self.prop = prop

        self.params = {}
        self.params.update(prop)

    def build_input_files(self, dataset, location, X):
        self.X = X

    def read_output(self):
        key = jr.key(123)
        key, subkey = jr.split(key)
        noise_p = jr.normal(subkey) * self.noise[0]
        key, subkey = jr.split(subkey)
        noise_s0 = jr.normal(key) * self.noise[1]
        key, subkey = jr.split(subkey)
        noise_s1 = jr.normal(key) * self.noise[1]

        U, V = self.geo["U"], self.geo["V"]
        eta, zeta = self.prop["shear"], self.prop["bulk"]

        X = self.X
        tau_bot = stress_bottom(X[:3], X[3:6], U, V, eta, zeta, X[6]) + noise_s0
        tau_top = stress_top(X[:3], X[3:6], U, V, eta, zeta, X[6]) + noise_s1
        press = eos_pressure(X[0:1], self.prop) + noise_p

        Y = jnp.hstack([press, tau_bot, tau_top]).T
        Ye = jnp.array([
            self.noise[0],  # p
            0., 0., 0.,  # xx, yy, zz
            self.noise[1], self.noise[1], 0.,  # yz, xz, xy
            0., 0., 0.,  # xx, yy, zz
            self.noise[1], self.noise[1], 0.  # yz, xz, xy
        ])

        return Y, Ye


class LennardJones(MolecularDynamics):

    name = 'lj'

    def __init__(self, params):
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

    name = 'mol'

    def __init__(self, params):
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

        # TODO: separate section of metadata
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
