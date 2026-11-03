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
import numpy as np
from scipy.signal import find_peaks
from scipy.constants import N_A


def _get_MPI_grid(Natoms, size, max_cpu, atoms_per_core=1000):
    """Estimate a suitable MPI processor grid.

    Parameters
    ----------
    Natoms : int
        Total number of atoms
    size : int
        Lateral size parameter
    max_cpu : int
        Maximum available processors
    atoms_per_core : float, optional
        Approximate minimum number of atoms per core (the default is 1000)

    Returns
    -------
    tuple
        Cartesian processor grid (int, int, int)
    """

    ncpus = min(max_cpu, Natoms // atoms_per_core)

    ny = size // 2 + size % 2
    if max_cpu < ny**2:
        ny = 1
        nx = 1
    else:
        nx = ny

    nz = max(ncpus // (nx * ny), 1)

    return (nx, ny, nz)


def _is_prime(n: int) -> bool:
    """Return True if n is a prime number, otherwise False."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    while i * 2 <= n:   # equivalent to i <= sqrt(n) but avoids importing math
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def sanitize_num_cpus(n, max_cpu):
    """
    Avoid prime number processor counts or numbers whose
    only factorization is (1, m, p) where m and p are prime.

    Parameters
    ----------
    n : int
        Estimated optimal number of processors

    Returns
    -------
    int
        Sanitized number whith reasonable factorization for MPI grid.
    """

    # n should be larger than 0
    n = max(1, n)

    # We may increase the number for more than double digit cpu counts
    if n + 1 <= max_cpu and n > 9:
        n += 1

    # but should not be larger than the max number requestes
    n = min(n, max_cpu)

    for m in range(n, 3, -1):

        if _is_prime(m):
            continue
        if m > 13 and m % 2 == 0 and _is_prime(m // 2):
            continue
        if m > 14 and m % 3 == 0 and _is_prime(m // 3):
            continue
        if m > 24 and m % 5 == 0 and _is_prime(m // 5):
            continue
        if m > 48 and m % 7 == 0 and _is_prime(m // 7):
            continue

        return m

    return n


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


def read_output_files_X(X):

    density_fluid_file = 'profile.dens.fluid.dat'
    density_solid_file = 'profile.dens.solid.dat'
    vx_fluid_file = 'profile.vx.fluid.dat'
    vy_fluid_file = 'profile.vy.fluid.dat'

    _, zdf, _, dens_f = np.loadtxt(density_fluid_file, unpack=True, skiprows=4)
    _, zds, _, dens_s = np.loadtxt(density_solid_file, unpack=True, skiprows=4)
    _, zvx, _, vx = np.loadtxt(vx_fluid_file, unpack=True, skiprows=4)
    _, zvy, _, vy = np.loadtxt(vy_fluid_file, unpack=True, skiprows=4)

    # Actual gap height (from density profile)
    zmin, zmax = _limits_from_nonzero(zdf, dens_f, zds, dens_s)

    gap_height = zmax - zmin

    # Actual fluid density
    mask = np.logical_and(zdf > zmin, zdf < zmax)
    density = np.mean(dens_f[mask]) * N_A * 1e-24  # from g/cm^3 to g/mol/A^3

    # Flux correction (from density and gap)
    j_fac = (density / X[0]) / (gap_height / X[3])

    # overwrite input data
    X = X.at[0].set(density)
    X = X.at[1].multiply(j_fac)
    X = X.at[2].multiply(j_fac)
    X = X.at[3].set(gap_height)

    return X


def _limits_from_peaks(z_fluid, dens_fluid, z_solid, dens_solid):

    # Outermost fluid maxima
    fluid_peaks, _ = find_peaks(dens_fluid)
    zmin_f = z_fluid[fluid_peaks[0]]
    zmax_f = z_fluid[fluid_peaks[-1]]

    # Innermost solid maxima
    nz = len(z_solid)
    solid_peaks_l, _ = find_peaks(dens_solid[:nz // 2])
    solid_peaks_u, _ = find_peaks(dens_solid[nz // 2:])
    zmin_s = z_solid[:nz // 2][solid_peaks_l[-1]]
    zmax_s = z_solid[nz // 2:][solid_peaks_u[0]]

    # Combined
    zmin = (zmin_s + zmin_f) / 2.
    zmax = (zmax_s + zmax_f) / 2.

    return zmin, zmax


def _limits_from_nonzero(z_fluid, dens_fluid, z_solid, dens_solid):

    # solid
    nz = len(z_solid)
    zsl = z_solid[:nz // 2]
    dsl = dens_solid[:nz // 2]
    zsu = z_solid[nz // 2:]
    dsu = dens_solid[nz // 2:]
    zmin_s = zsl[dsl / dsl.max() > 0.01].max()
    zmax_s = zsu[dsu / dsu.max() > 0.01].min()

    # fluid
    zmin_f = z_fluid[dens_fluid / dens_fluid.max() > 0.01].min()
    zmax_f = z_fluid[dens_fluid / dens_fluid.max() > 0.01].max()

    # Combined
    zmin = (zmin_s + zmin_f) / 2.
    zmax = (zmax_s + zmax_f) / 2.

    return zmin, zmax


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
        Statistical inefficiency parameter
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
