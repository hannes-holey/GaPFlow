import numpy as np
import os

import os
import sys
from mpi4py import MPI
try:
    from lammps import lammps
except ImportError:
    pass


def main():

    comm = MPI.Comm.Get_parent()

    run_serial()

    comm.Barrier()
    comm.Free()


def run_parallel(nworker):

    worker_file = os.path.abspath(__file__)

    sub_comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[worker_file],
                                   maxprocs=nworker)

    # Parameter broadcasting fails on some systems
    # kw_args = sub_comm.bcast(kw_args, root=0)

    # Wait for MD to complete and free spawned communicator
    sub_comm.Barrier()
    sub_comm.Free()


def run_serial(system='lj'):

    nargs = ["-log", "log.lammps"]

    lmp = lammps(cmdargs=nargs)

    assert lmp.has_package('EXTRA-FIX'), "Lammps needs to be compiled with package 'EXTRA-FIX'"

    if system == 'lj':
        # Invoke parameters and wall definition
        lmp.command("include in.param")
        lmp.command("variable slabfile index in.wall")
        lmp.file("in.run")
    elif system == 'mol':
        lmp.file("run.in.all")


if __name__ == "__main__":
    # main is called by an individual spawned process for parallel MD runs
    main()


def write_input_files(X, proto_ds, md):

    # LJ system

    num_cpu = md['ncpu']

    # write variables file (# FIXME: height, indices not hardcoded)
    variables_str = f"""
variable    input_gap equal {X[0]}
variable    input_dens equal {X[3]}
variable    input_fluxX equal {X[4]}
variable    input_fluxY equal {X[5]}
"""

    # if md['mode'] == 'slip'
    #     variables_str += f'variable input_kappa equal {X[?]}\n'

    excluded = ['infile', 'wallfile', 'ncpu']
    for k, v in md.items():
        if k not in excluded:
            variables_str += f'variable {k} equal {v}\n'

    with open(os.path.join('in.param'), 'w') as f:
        f.writelines(variables_str)

    # Move inputfiles to proto dataset
    proto_ds.put_item(md['wallfile'], 'in.wall')
    proto_ds.put_item(md['infile'], 'in.run')
    proto_ds.put_item('in.param', 'in.param')
    os.remove('in.param')

    # Gold / alkane system

    # args = self.md

    # # Move inputfiles to proto dataset
    # os.makedirs(os.path.join("moltemplate_files"))
    # os.makedirs(os.path.join("static"))

    # proto_ds.put_item(self.md['fftemplate'],
    #                   os.path.join("moltemplate_files", os.path.basename(self.md['fftemplate'])))
    # proto_ds.put_item(self.md['topo'],
    #                   os.path.join("moltemplate_files", os.path.basename(self.md['topo'])))

    # for f in os.listdir(self.md["staticFiles"]):
    #     proto_ds.put_item(os.path.join(self.md["staticFiles"], f), os.path.join("static", f))

    # # TODO: separate section of metadata

    # args["density"] = float(Xnew[-3, index])
    # args["fluxX"] = float(Xnew[-2, index])
    # args["fluxY"] = float(Xnew[-1, index])

    # if self.mode.startswith('gap'):
    #     args["gap_height"] = float(Xnew[0, index])

    # if self.mode == "gap_angle":
    #     args["rotation"] = float(Xnew[1, index])

    # num_cpu = write_template(args)
    # build_template(args)


def read_output_files():
    # Get stress
    # Apply unit conversion from LAMMPS output to READMEs
    md_data = np.loadtxt('stress_wall.dat')  # * self._stress_scale

    Y = np.zeros((13,))
    Yerr = np.zeros((13,))

    if md_data.shape[1] == 5:
        # 1D
        # step, pL_t, tauL_t, pU_t, tauU_t = np.loadtxt('stress_wall.dat', unpack=True)

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
        Yerr[0] = (pL_err + pU_err) / 2.
        Yerr[5] = tauxzL_err
        Yerr[11] = tauxzU_err

    elif md_data.shape[1] == 7:
        # 2D
        # step, pL_t, tauxzL_t, pU_t, tauxzU_t, tauyzL_t, tauyzU_t = np.loadtxt('stress_wall.dat', unpack=True)

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
        Yerr[0] = (pL_err + pU_err) / 2.
        Yerr[4] = tauyzL_err
        Yerr[5] = tauxzL_err
        Yerr[10] = tauyzU_err
        Yerr[11] = tauxzU_err

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
