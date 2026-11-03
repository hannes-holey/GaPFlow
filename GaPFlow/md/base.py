#
# Copyright 2025-2026 Hannes Holey
#           2026 Dan Waxman
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
from .runner import run_parallel, run_serial
from ..db import README_SCHEMA_VERSION
from ..utils import make_dumpable
from ..logging import get_logger

import os
import abc
import dtoolcore
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from ruamel.yaml import YAML
from getpass import getuser
from urllib.parse import urlparse


yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=4, sequence=4, offset=2)

logger = get_logger("gapflow.md")


class MolecularDynamics:
    """Driver for molecular dynamics simulations.

    Abstract base class for MD setup, running, and reading outputs.
    Derived classes need to implement methods to write LAMMPS input files
    into a dtool dataset, and to read the output of this simulation.

    Attributes
    ----------
    name : str
        Name of the MD object
    params : dict
        Parameters to control the MD setup, will be written to the dtool metadata.
    main_file : str
        File name of the main LAMMPS input file.
    num_worker : int
        Number of cores to run the parallel MD simulation.
    is_mock : bool
        Whether the subclass is only a mock object, which does not run an actual MD simulation.
    """
    __metaclass__ = abc.ABCMeta

    name = str
    params: dict
    main_file: str
    num_worker: int
    is_mock: bool
    _dtool_basepath: str = '/tmp/'
    _readme_template: str = ""
    _input_names: list[str] = ['ρ', 'jx', 'jy', 'h', '∂h/∂x', '∂h/∂y', 'U', 'V'] + [f'extra_{i}' for i in range(10)]
    _output_names: list[str] = ['p',
                                'τ(xx|bot)', 'τ(yy|bot)', 'τ(zz|bot)', 'τ(yz|bot)', 'τ(xz|bot)', 'τ(xy|bot)',
                                'τ(xx|top)', 'τ(yy|top)', 'τ(zz|top)', 'τ(yz|top)', 'τ(xz|top)', 'τ(xy|top)']
    _ascii_art: str = r"""
  _        _    __  __ __  __ ____  ____
 | |      / \  |  \/  |  \/  |  _ \/ ___|
 | |     / _ \ | |\/| | |\/| | |_) \___ \
 | |___ / ___ \| |  | | |  | |  __/ ___) |
 |_____/_/   \_\_|  |_|_|  |_|_|   |____/

"""

    @property
    def dtool_basepath(self):
        """File location, where dtool datasets are written into (default is '/tmp/')."""
        return self._dtool_basepath

    @dtool_basepath.setter
    def dtool_basepath(self, name):
        self._dtool_basepath = name

    @abc.abstractmethod
    def build_input_files(self, dataset, location, X) -> None:
        """Builds LAMMPS input files based on GP inputs
         and writes them to a dtool dataset.

        Parameters
        ----------
        dataset : dtoolcore.proto_dataset
            A proto_dataset object.
        location : str
            Absolute path of the proto dataset.
        X : Array
            Input (i.e. density, gap height, ...)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def read_output(self):
        """Read simulation output and returns observations and their standard error."""
        raise NotImplementedError

    def _write_log_input(self, proto_datapath, X) -> None:
        """Write logging info before an MD run.

        Parameters
        ----------
        proto_datapath : str
            The data path inside the dtool proto dataset
        X : Array
            The target input array
        """

        logger.info(120 * '=')
        logger.info('Running MD simulation')
        logger.info('---')
        logger.info(self._ascii_art)
        logger.info('---')

        for i, (Xi, name) in enumerate(zip(X, self._input_names)):
            logger.info(f'Target input {i + 1}: {Xi:+.3e}    ({name})')

        logger.info('---')
        logger.info('View log:')
        logger.info(f'{proto_datapath}/data/log.lammps')
        logger.info('---')

    def _write_log_output(self, X, X_target, Y, Ye, walltime) -> None:
        """Write logging info after an MD run.

        Parameters
        ----------
        X : Array
            The realized input
        X_target : Array
            The target input
        Y : Array
            Measured output
        Ye : Array
            Measured output error
        walltime: datetime.timedelta
            Wall time of an MD run
        """

        logger.info('---')
        logger.info(f'MD run completed (walltime: {str(walltime).split(".")[0]})')
        logger.info('---')

        for i, (Xi, Xi_t, name) in enumerate(zip(X, X_target, self._input_names)):

            if abs(Xi_t) < 1e-9:
                logger.info(f'Measured input {i + 1}: {Xi:+.3e} (----%)   ({name})')
            else:
                deviation = (Xi - Xi_t) / abs(Xi_t) * 100
                logger.info(f'Measured input {i + 1}: {Xi:+.3e} ({deviation:+.1f}%)   ({name})')

        logger.info('---')

        for i, (Yi, Yie, name) in enumerate(zip(Y, Ye, self._output_names)):
            logger.info(f'Measured output {i + 1:2d}: {Yi:+.3e} ±{Yie:.3e}  ({name})')
            if i in [0, 6]:
                logger.info('')

        logger.info(120 * '=')

    def _write_dtool_readme(self, dataset_path, Xnew, Ynew, Yerrnew):
        """Write the simulation metadata into the dtool README.

        Parameters
        ----------
        dataset_path : str
            Path of the dtool dataset.
        Xnew : Array
            New inputs.
        Ynew : Array
            New observations (from MD).
        Yerrnew : [type]
            New observatio standard error (from MD).
        """
        if len(self._readme_template) == 0:
            metadata = {}
        else:
            metadata = yaml.load(self._readme_template)

        # Update metadata
        metadata["owners"] = [{'username': getuser()}]
        metadata["creation_date"] = date.today()
        metadata["expiration_date"] = metadata["creation_date"] + relativedelta(years=10)
        metadata["schema_version"] = README_SCHEMA_VERSION

        out_fname = os.path.join(dataset_path, 'README.yml')

        metadata.update({'parameters': make_dumpable(self.params)})

        metadata['X'] = make_dumpable(Xnew)
        metadata['Y'] = make_dumpable(Ynew)
        metadata['Yerr'] = make_dumpable(Yerrnew)

        with open(out_fname, 'w') as outfile:
            yaml.dump(metadata, outfile)

    def _create_dtool_dataset(self, tag):
        """Create a dtool proto dataset. The name of the dataset consists of a time stamp,
        the name of the MD runner, and a tag (e.g. a number).

        Parameters
        ----------
        tag : str
            A tag to attach to the dataset name.

        Returns
        -------
        dtoolcore.proto_dataset
            The proto_dataset object
        str
            Current path to the dataset
        """
        ds_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self.name}-{tag:03}'

        proto_ds = dtoolcore.create_proto_dataset(name=ds_name,
                                                  base_uri=self.dtool_basepath)

        proto_ds_path = urlparse(proto_ds.uri).path

        if os.name == 'nt':
            proto_ds_path = proto_ds_path[1:]

        return proto_ds, proto_ds_path

    def run(self, X_target, tag):
        """Run an MD simulation and store its input, metadata, and output into a dtool dataset.

        This method is called from a Database instance when new training data is added e.g. during
        initialization or in an active learning simulation.

        Parameters
        ----------
        X_target : Array
            The training input.
        tag : str
            A tag to attach to the dataset name.

        Returns
        -------
        Array
            Training observations
        Array
            Standard error of training observations
        """

        # Setup MD simulation
        dataset, location = self._create_dtool_dataset(tag)
        self.build_input_files(dataset, location, X_target)

        self._write_log_input(location, X_target)

        # Move to dtool datapath...
        basedir = os.getcwd()
        os.chdir(os.path.join(location, 'data'))

        tic = datetime.now()

        # ...Run MD...
        if self.num_worker > 1:
            run_parallel(self.main_file, self.num_worker)
        elif self.num_worker == 1:
            run_serial(self.main_file)
        else:
            pass

        toc = datetime.now()

        # ...Read output / post-process MD result...
        X, Y, Ye = self.read_output()

        # ...and return to cwd
        os.chdir(basedir)

        # Finalize dataset
        walltime = toc - tic
        self._write_log_output(X, X_target, Y, Ye, walltime)
        self._write_dtool_readme(location, X, Y, Ye)
        dataset.freeze()

        return X, Y, Ye
