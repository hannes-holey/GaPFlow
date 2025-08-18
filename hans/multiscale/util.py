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
