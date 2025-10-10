import os
import sys
import abc
import dtoolcore
from mpi4py import MPI
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from ruamel.yaml import YAML
from getpass import getuser
from socket import gethostname
import scipy.constants as sci

try:
    from lammps import lammps
except ImportError:
    pass

import jax.random as jr
import jax.numpy as jnp
import numpy as np

from GaPFlow.models.pressure import eos_pressure
from GaPFlow.models.viscous import stress_bottom, stress_top
from GaPFlow.moltemplate.main import write_template, build_template
from GaPFlow.utils import bordered_text

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
    lmp = lammps(cmdargs=nargs)
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
    _dtool_basepath: str
    readme_template: str = ""
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
        for i, Xi in enumerate(X):
            text.append(f'Input {i + 1}: {Xi:.3g}')
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

        X = [float(item) for item in np.asarray(Xnew)]
        Y = [float(item) for item in np.asarray(Ynew)]
        Yerr = [float(item) for item in np.asarray(Yerrnew)]

        metadata.update({'parameters': self.params})

        metadata['X'] = X
        metadata['Y'] = Y
        metadata['Yerr'] = Yerr

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
        noise_p = jr.normal(key) * self.noise[0]
        noise_s0 = jr.normal(key) * self.noise[1]
        noise_s1 = jr.normal(key) * self.noise[1]

        U, V = self.geo["U"], self.geo["V"]
        eta, zeta = self.prop["shear"], self.prop["bulk"]

        X = self.X
        tau_bot = stress_bottom(X[3:], X[:3], U, V, eta, zeta, 0.0) + noise_s0
        tau_top = stress_top(X[3:], X[:3], U, V, eta, zeta, 0.0) + noise_s1
        press = eos_pressure(X[3:4], self.prop) + noise_p

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
        self.main_file = 'in.run'
        self.num_worker = params['ncpu']
        self.params = params

    def build_input_files(self, dataset, location, X):
        # write variables file
        variables_str = f"""
variable\tinput_gap equal {X[0]}
variable\tinput_dens equal {X[3]}
variable\tinput_fluxX equal {X[4]}
variable\tinput_fluxY equal {X[5]}
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
        args = self.params
        args["gap_height"] = float(X[0])
        args["density"] = float(X[3])
        args["fluxX"] = float(X[4])
        args["fluxY"] = float(X[5])

        cwd = os.getcwd()
        os.chdir(proto_ds_datapath)
        self.num_worker = write_template(args)
        build_template(args)
        os.chdir(cwd)

    def read_output(self):
        sf = sci.calorie * 1e-4  # from kcal/mol/A^3 to g/mol/A/fs^2
        return read_output_files(sf=sf)


def read_output_files(fname='stress_wall.dat', sf=1.):

    md_data = np.loadtxt(fname) * sf

    Y = np.zeros((13,))
    Yerr = np.zeros((13,))

    if md_data.shape[1] == 5:
        # 1D
        # timeseries
        pressL_t = md_data[:, 1]
        pressU_t = md_data[:, 3]
        tauL_t = md_data[:, 2]
        tauU_t = md_data[:, 4]

        # mean
        pressL = np.mean(pressL_t)
        pressU = np.mean(pressU_t)
        tauL = np.mean(tauL_t)
        tauU = np.mean(tauU_t)

        # variance of mean
        pL_err = variance_of_mean(pressL_t)
        pU_err = variance_of_mean(pressU_t)
        tauxzL_err = variance_of_mean(tauL_t)
        tauxzU_err = variance_of_mean(tauU_t)

        # fill into buffer
        Y[0] = (pressL + pressU) / 2.
        Y[5] = tauL
        Y[11] = tauU
        Yerr[0] = np.sqrt((pL_err + pU_err) / 2.)
        Yerr[5] = np.sqrt(tauxzL_err)
        Yerr[11] = np.sqrt(tauxzU_err)

    elif md_data.shape[1] == 7:
        # 2D
        # timeseries data
        pressL_t = md_data[:, 1]
        pressU_t = md_data[:, 3]
        tauxzL_t = md_data[:, 2]
        tauxzU_t = md_data[:, 4]
        tauyzL_t = md_data[:, 5]
        tauyzU_t = md_data[:, 6]

        # mean
        pressL = np.mean(pressL_t)
        pressU = np.mean(pressU_t)
        tauxzL = np.mean(tauxzL_t)
        tauxzU = np.mean(tauxzU_t)
        tauyzL = np.mean(tauyzL_t)
        tauyzU = np.mean(tauyzU_t)

        # variance of mean
        pL_err = variance_of_mean(pressL_t)
        pU_err = variance_of_mean(pressU_t)
        tauxzL_err = variance_of_mean(tauxzL_t)
        tauxzU_err = variance_of_mean(tauxzU_t)
        tauyzL_err = variance_of_mean(tauyzL_t)
        tauyzU_err = variance_of_mean(tauyzU_t)

        # fill into buffer
        Y[0] = (pressL + pressU) / 2.
        Y[4] = tauyzL
        Y[5] = tauxzL
        Y[10] = tauyzU
        Y[11] = tauxzU
        Yerr[0] = np.sqrt((pL_err + pU_err) / 2.)
        Yerr[4] = np.sqrt(tauyzL_err)
        Yerr[5] = np.sqrt(tauxzL_err)
        Yerr[10] = np.sqrt(tauyzU_err)
        Yerr[11] = np.sqrt(tauxzU_err)

    return Y, Yerr


def autocorr_func_1d(x):
    """

    Compute autocorrelation function of 1D time series.

    Parameters
    ----------
    x : numpy.ndarry
        The time series

    Returns
    -------
    numpy.ndarray
        Normalized time autocorrelation function
    """
    n = len(x)

    # unbias
    x -= np.mean(x)

    # pad with zeros
    ext_size = 2 * n - 1
    fsize = 2**np.ceil(np.log2(ext_size)).astype('int')

    # ACF from FFT
    x_f = np.fft.fft(x, fsize)
    C = np.fft.ifft(x_f * x_f.conjugate())[:n] / (n - np.arange(n))

    # Normalize
    C_t = C.real / C.real[0]

    return C_t


def statistical_inefficiency(timeseries, mintime):
    """
    see e.g. Chodera et al. J. Chem. Theory Comput., Vol. 3, No. 1, 2007

    Parameters
    ----------
    timeseries : numpy.ndarray
        The time series
    mintime : int
        Minimum time lag to calculate correlation time

    Returns
    -------
    float
        Statisitical inefficiency parameter
    """
    N = len(timeseries)
    C_t = autocorr_func_1d(timeseries)
    t_grid = np.arange(N).astype('float')
    g_t = 2.0 * C_t * (1.0 - t_grid / float(N))
    ind = np.where((C_t <= 0) & (t_grid > mintime))[0][0]
    g = 1.0 + g_t[1:ind].sum()
    return max(1.0, g)


def variance_of_mean(timeseries, mintime=1):
    """

    Compute the variance of the mean value for a correlated time series

    Parameters
    ----------
    timeseries : numpy.ndarray
        The time series
    mintime : int, optional
        Minimum time lag to calculate correlation time

    Returns
    -------
    float
        Variance of the mean
    """

    g = statistical_inefficiency(timeseries, mintime)
    n = len(timeseries)
    var = np.var(timeseries) / n * g

    return var
