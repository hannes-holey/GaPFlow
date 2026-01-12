#
# Copyright 2025 Christoph Huber
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
"""Analytic solutions for validation of FEM implementations."""

import numpy as np


def heat_equation_1d(L, c_v, k, rho, T_max, x_vec, t_vec):
    """Analytical solution for the 1D heat equation.

    Solves the heat equation with Dirichlet T=0 boundary conditions
    at both ends (x=0 and x=L) and an initial condition of
    T(x,0) = T_max * sin(pi*x/L).

    The solution is:
        T(x,t) = T_max * exp(-lambda^2 * t) * sin(pi*x/L)

    where lambda = sqrt(alpha) * pi/L and alpha = k/(c_v*rho).

    Parameters
    ----------
    L : float
        Domain length
    c_v : float
        Specific heat capacity
    k : float
        Thermal conductivity
    rho : float
        Density
    T_max : float
        Maximum initial temperature (amplitude of sine wave)
    x_vec : array_like
        Spatial coordinates, shape (nx,)
    t_vec : array_like
        Time values, shape (nt,)

    Returns
    -------
    T_mat : ndarray
        Temperature field, shape (nt, nx), where T_mat[i, j] = T(x_vec[j], t_vec[i])
    """
    mu = np.pi / L
    alpha = k / (c_v * rho)
    lambda_ = np.sqrt(alpha) * mu

    def T_func(x, t):
        return np.exp(-lambda_**2 * t) * np.sin(mu * x)

    T_mat = T_func(x_vec[None, :], t_vec[:, None]) * T_max
    return T_mat
