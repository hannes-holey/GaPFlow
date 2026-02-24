#
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
from ..utils import bordered_text, make_dumpable

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
    _input_names: list[str] = ['ρ', 'jx', 'jy', 'h', '∂h/∂x', '∂h/∂y'] + [f'extra_{i}' for i in range(10)]
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
        """Builds LAMMPS input files based on GP inputs and writes them to a dtool dataset.

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
        """Read simulation output and return observations and their standard error."""
        raise NotImplementedError

    def _pretty_print(self, proto_datapath, X) -> None:
        """Print header before the start of a LAMMPS simulation.

        Parameters
        ----------
        proto_datapath : str
            The data path inside the dtool proto dataset
        X : Array
            The input array
        """
        text = ['Run next MD simulation in:', f'{proto_datapath}']
        text.append(self._ascii_art)
        text.append('---')
        for i, (Xi, name) in enumerate(zip(X, self._input_names)):
            text.append(f'Input {i + 1}: {Xi:+.3e}    ({name})')
        print(bordered_text('\n'.join(text)))

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

    def run(self, X, tag):
        """Run an MD simulation and store its input, metadata, and output into a dtool dataset.

        This method is called from a Database instance when new training data is added e.g. during
        initialization or in an active learning simulation.

        Parameters
        ----------
        X : Array
            The training input.
        tag : str
            A tag to attach to the dataset name.

        Returns
        -------
        Array
            Training observations
        Array
            Standard error of traiing observations
        """

        # Setup MD simulation
        dataset, location = self._create_dtool_dataset(tag)
        self.build_input_files(dataset, location, X)

        self._pretty_print(location, X)

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
        self._write_dtool_readme(location, X, Y, Ye)
        dataset.freeze()

        return Y, Ye
