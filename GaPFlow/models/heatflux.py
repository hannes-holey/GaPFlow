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
import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[np.floating]


def heatflux_top(
    h: float,
    h_eff: float,
    k: float,
    c_v: float,
    eta: float,
    rho: float,
    E: float,
    jx: float,
    jy: float,
    U: float,
    V: float,
    T_bulk_top: float,
    T_bulk_bot: float,
    A: float
) -> float:
    """
    Heat flux fluid -> top wall using Robin BC

    Parameters:
    ------------
        h: channel height [m]
        h_eff: effective heat transfer coefficient [m^2K/W]
        k: thermal conductivity [W/mK]
        c_v: specific heat capacity [J/kgK]
        eta: dynamic viscosity [Pa s]
        rho: density [kg/m^3]
        E: total energy per unit volume [J/m^3]
        jx: momentum flux in x direction [kg/m^2s]
        jy: momentum flux in y direction [kg/m^2s]
        U: bottom wall velocity in x direction [m/s]
        V: bottom wall velocity in y direction [m/s]
        T_bulk_top: bulk temperature top wall [K]
        T_bulk_bot: bulk temperature bottom wall [K]
        A: wall area [m^2]

    Returns:
    --------
        float: heat flux at the top wall [W/m^2]
    """
    # Precompute common terms
    h2 = h * h
    rho2 = rho * rho
    jx2 = jx * jx
    jy2 = jy * jy
    U2 = U * U
    V2 = V * V
    k2 = k * k
    h_eff2 = h_eff * h_eff

    # Denominator: 500*c_v*k*rho^2*(h^2 + 8*h*h_eff*k + 12*h_eff^2*k^2)
    denom = 500 * c_v * k * rho2 * (h2 + 8 * h * h_eff * k + 12 * h_eff2 * k2)

    # Numerator from MATLAB expression
    numerator = (
        30 * h_eff * jx2 * k2 + 30 * h_eff * jy2 * k2
        + 15 * h * jx2 * k + 15 * h * jy2 * k
        - 60 * E * h_eff * k2 * rho
        - 12 * c_v * eta * h * jx2 - 12 * c_v * eta * h * jy2
        - 30 * E * h * k * rho
        + 2 * U2 * c_v * eta * h * rho2 + 2 * V2 * c_v * eta * h * rho2
        + 60 * T_bulk_top * c_v * h_eff * k2 * rho2
        + 10 * T_bulk_bot * c_v * h * k * rho2 + 20 * T_bulk_top * c_v * h * k * rho2
        - 24 * c_v * eta * h_eff * jx2 * k - 24 * c_v * eta * h_eff * jy2 * k
        + 24 * U2 * c_v * eta * h_eff * k * rho2 + 24 * V2 * c_v * eta * h_eff * k * rho2
        + 2 * U * c_v * eta * h * jx * rho + 2 * V * c_v * eta * h * jy * rho
        - 36 * U * c_v * eta * h_eff * jx * k * rho - 36 * V * c_v * eta * h_eff * jy * k * rho
    )

    q_top = -numerator / denom
    return q_top


def heatflux_bot(
    h: float,
    h_eff: float,
    k: float,
    c_v: float,
    eta: float,
    rho: float,
    E: float,
    jx: float,
    jy: float,
    U: float,
    V: float,
    T_bulk_top: float,
    T_bulk_bot: float,
    A: float
) -> float:
    """
    Heat flux fluid -> bottom wall using Robin BC

    Parameters:
    ------------
        h: channel height [m]
        h_eff: effective heat transfer coefficient [m^2K/W]
        k: thermal conductivity [W/mK]
        c_v: specific heat capacity [J/kgK]
        eta: dynamic viscosity [Pa s]
        rho: density [kg/m^3]
        E: total energy per unit volume [J/m^3]
        jx: momentum flux in x direction [kg/m^2s]
        jy: momentum flux in y direction [kg/m^2s]
        U: bottom wall velocity in x direction [m/s]
        V: bottom wall velocity in y direction [m/s]
        T_bulk_top: bulk temperature top wall [K]
        T_bulk_bot: bulk temperature bottom wall [K]
        A: wall area [m^2]

    Returns:
    --------
        float: heat flux at the bottom wall [W/m^2]
    """
    # Precompute common terms
    h2 = h * h
    h4 = h2 * h2
    h5 = h4 * h
    rho2 = rho * rho
    jx2 = jx * jx
    jy2 = jy * jy
    U2 = U * U
    V2 = V * V
    k2 = k * k

    # Denominator: 500*c_v*h^4*k*rho^2*(h + 2*h_eff*k)*(h + 6*h_eff*k)
    denom = 500 * c_v * h4 * k * rho2 * (h + 2 * h_eff * k) * (h + 6 * h_eff * k)

    # Numerator from MATLAB expression (with h^5 and h^4 terms)
    numerator = (
        15 * h5 * jx2 * k + 15 * h5 * jy2 * k
        + 30 * h4 * h_eff * jx2 * k2 + 30 * h4 * h_eff * jy2 * k2
        - 30 * E * h5 * k * rho
        - 12 * c_v * eta * h5 * jx2 - 12 * c_v * eta * h5 * jy2
        - 60 * E * h4 * h_eff * k2 * rho
        + 20 * T_bulk_bot * c_v * h5 * k * rho2 + 10 * T_bulk_top * c_v * h5 * k * rho2
        - 8 * U2 * c_v * eta * h5 * rho2 - 8 * V2 * c_v * eta * h5 * rho2
        + 22 * U * c_v * eta * h5 * jx * rho + 22 * V * c_v * eta * h5 * jy * rho
        - 24 * c_v * eta * h4 * h_eff * jx2 * k - 24 * c_v * eta * h4 * h_eff * jy2 * k
        + 60 * T_bulk_bot * c_v * h4 * h_eff * k2 * rho2
        - 36 * U2 * c_v * eta * h4 * h_eff * k * rho2 - 36 * V2 * c_v * eta * h4 * h_eff * k * rho2
        + 84 * U * c_v * eta * h4 * h_eff * jx * k * rho + 84 * V * c_v * eta * h4 * h_eff * jy * k * rho
    )

    q_bot = -numerator / denom
    return q_bot


def get_T_z(
    h: float,
    h_eff: float,
    k: float,
    c_v: float,
    eta: float,
    rho: float,
    E: float,
    jx: float,
    jy: float,
    U: float,
    V: float,
    T_bulk_top: float,
    T_bulk_bot: float,
    z: NDArray
) -> NDArray:
    """Calculate temperature profile in the channel using 2nd order velocity profile

    Parameters:
    ------------
        h: channel height [m]
        h_eff_inv: effective heat transfer coefficient [m^2K/W]
        k: thermal conductivity [W/mK]
        c_v: specific heat capacity [J/kgK]
        eta: dynamic viscosity [Pa s]
        rho: density [kg/m^3]
        E: total energy per unit volume [J/m^3]
        jx: momentum flux in x direction [kg/m^2s]
        jy: momentum flux in y direction [kg/m^2s]
        U: bottom wall velocity in x direction [m/s]
        V: bottom wall velocity in y direction [m/s]
        T_bulk_top: bulk temperature top wall [K]
        T_bulk_bot: bulk temperature bottom wall [K]
        z: positions within the channel [m]

    Returns:
        NDArray: temperature at position z
    """
    assert np.all((z >= 0) & (z <= h)), "z positions must be within the channel height [0, h]"

    # Precompute common terms
    h2 = h * h
    h3 = h2 * h
    h4 = h3 * h
    rho2 = rho * rho
    jx2 = jx * jx
    jy2 = jy * jy
    U2 = U * U
    V2 = V * V
    h_eff2 = h_eff * h_eff
    k2 = k * k
    z2 = z * z
    z3 = z2 * z
    z4 = z3 * z

    denom1 = 5 * c_v * rho2 * (h2 + 8 * h * h_eff * k + 12 * h_eff2 * k2)

    term1_num = (
        60 * E * h_eff2 * k2 * rho - 30 * h_eff2 * jy2 * k2 - 15 * h * h_eff * jx2 * k
        - 15 * h * h_eff * jy2 * k
        - 30 * h_eff2 * jx2 * k2 + 5 * T_bulk_bot * c_v * h2 * rho2
        + 24 * c_v * eta * h_eff2 * jx2 * k + 24 * c_v * eta * h_eff2 * jy2 * k
        + 30 * E * h * h_eff * k * rho + 12 * c_v * eta * h * h_eff * jx2
        + 12 * c_v * eta * h * h_eff * jy2
        + 20 * T_bulk_bot * c_v * h * h_eff * k * rho2
        - 10 * T_bulk_top * c_v * h * h_eff * k * rho2
        + 8 * U2 * c_v * eta * h * h_eff * rho2 + 8 * V2 * c_v * eta * h * h_eff * rho2
        + 36 * U2 * c_v * eta * h_eff2 * k * rho2 + 36 * V2 * c_v * eta * h_eff2 * k * rho2
        - 22 * U * c_v * eta * h * h_eff * jx * rho
        - 22 * V * c_v * eta * h * h_eff * jy * rho
        - 84 * U * c_v * eta * h_eff2 * jx * k * rho
        - 84 * V * c_v * eta * h_eff2 * jy * k * rho
    )
    term1 = (k * term1_num) / denom1

    term2_num = (
        30 * h_eff * jx2 * k2 + 30 * h_eff * jy2 * k2 + 15 * h * jx2 * k + 15 * h * jy2 * k
        - 60 * E * h_eff * k2 * rho - 12 * c_v * eta * h * jx2 - 12 * c_v * eta * h * jy2
        - 30 * E * h * k * rho - 8 * U2 * c_v * eta * h * rho2 - 8 * V2 * c_v * eta * h * rho2
        + 60 * T_bulk_bot * c_v * h_eff * k2 * rho2 + 20 * T_bulk_bot * c_v * h * k * rho2
        + 10 * T_bulk_top * c_v * h * k * rho2 - 24 * c_v * eta * h_eff * jx2 * k
        - 24 * c_v * eta * h_eff * jy2 * k
        - 36 * U2 * c_v * eta * h_eff * k * rho2 - 36 * V2 * c_v * eta * h_eff * k * rho2
        + 22 * U * c_v * eta * h * jx * rho + 22 * V * c_v * eta * h * jy * rho
        + 84 * U * c_v * eta * h_eff * jx * k * rho + 84 * V * c_v * eta * h_eff * jy * k * rho
    )
    term2 = (z * term2_num) / denom1

    vel_term = U2 * rho2 - 4 * U * jx * rho + V2 * rho2 - 4 * V * jy * rho + 4 * jx2 + 4 * jy2
    term3 = (3 * eta * z4 * vel_term) / (h4 * rho2)

    vel_term2 = 2 * U2 * rho2 - 7 * U * jx * rho + 2 * V2 * rho2 - 7 * V * jy * rho + 6 * jx2 + 6 * jy2
    term4 = (4 * eta * z3 * vel_term2) / (h3 * rho2)

    vel_term3 = 4 * U2 * rho2 - 12 * U * jx * rho + 4 * V2 * rho2 - 12 * V * jy * rho + 9 * jx2 + 9 * jy2
    term5 = (2 * eta * z2 * vel_term3) / (h2 * rho2)

    term6_num = (
        15 * h * jx2 * k + 15 * h * jy2 * k + 18 * c_v * eta * h * jx2 + 18 * c_v * eta * h * jy2
        - 30 * E * h * k * rho + 7 * U2 * c_v * eta * h * rho2 + 7 * V2 * c_v * eta * h * rho2
        + 15 * T_bulk_bot * c_v * h * k * rho2 + 15 * T_bulk_top * c_v * h * k * rho2
        + 180 * c_v * eta * h_eff * jx2 * k + 180 * c_v * eta * h_eff * jy2 * k
        + 60 * U2 * c_v * eta * h_eff * k * rho2 + 60 * V2 * c_v * eta * h_eff * k * rho2
        - 18 * U * c_v * eta * h * jx * rho - 18 * V * c_v * eta * h * jy * rho
        - 180 * U * c_v * eta * h_eff * jx * k * rho - 180 * V * c_v * eta * h_eff * jy * k * rho
    )
    term6 = (z2 * term6_num) / (5 * c_v * h * rho2 * (h2 + 6 * h_eff * k * h))

    T_vec = (term1 - term2 - term3 + term4 - term5 + term6) / k
    return T_vec


if __name__ == "__main__":
    # Example usage with flow
    print("=" * 60)
    print("TEST CASE 1: With flow (Couette-like)")
    print("=" * 60)

    h = 1.  # channel height in meters
    h_eff = 1.  # Robin BC parameter
    k = 0.01  # thermal conductivity in W/(m·K)
    c_v = 1  # specific heat capacity at constant volume in J/(kg·K)
    eta = 1e-02  # dynamic viscosity in Pa·s
    rho = 1  # density in kg/m³
    T_avg = 300  # average temperature in K
    jx = 10  # momentum in x direction per unit volume
    jy = 0  # momentum in y direction per unit volume
    U = 10  # velocity in x direction at bottom wall
    V = 0  # velocity in y direction at bottom wall
    T_bulk_top = 300  # bulk temperature at top wall in K
    T_bulk_bot = 300  # bulk temperature at bottom wall in K

    E = T_avg * rho * c_v + 0.5 * (jx**2 + jy**2) / rho

    print(f"\nInput parameters:")
    print(f"  h = {h}, h_eff = {h_eff}, k = {k}, cv = {c_v}")
    print(f"  eta = {eta}, rho = {rho}")
    print(f"  jx = {jx}, U = {U}")
    print(f"  T_bulk_top = {T_bulk_top}, T_bulk_bot = {T_bulk_bot}")
    print(f"  E = {E}")

    q_top = heatflux_top(h, h_eff, k, c_v, eta, rho, E, jx, jy, U, V, T_bulk_top, T_bulk_bot, None)
    q_bot = heatflux_bot(h, h_eff, k, c_v, eta, rho, E, jx, jy, U, V, T_bulk_top, T_bulk_bot, None)
    print(f"\nResults:")
    print(f"  q_top = {q_top:.6e} W/m²")
    print(f"  q_bot = {q_bot:.6e} W/m²")

    z_positions = np.linspace(0, h, 20)
    T_vec = get_T_z(h, h_eff, k, c_v, eta, rho, E, jx, jy, U, V, T_bulk_top, T_bulk_bot, z_positions)
    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.plot(T_vec, z_positions)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Position in channel (m)")
    plt.title("Temperature profile in the channel (with flow)")
    plt.grid()
    plt.show()

    # Test case: No flow, symmetric walls - checking for asymmetry bug
    print("\n" + "=" * 60)
    print("TEST CASE: No flow, symmetric walls (asymmetry check)")
    print("=" * 60)

    # Parameters matching the FEM tutorial use case
    h_test = 1.0e-5       # channel height [m]
    h_eff_test = 1.0e4    # Robin BC parameter (thermal resistance)
    k_test = 0.01        # thermal conductivity [W/(m·K)]
    cv_test = 1000.      # specific heat capacity [J/(kg·K)]
    eta_test = 1.846e-5   # dynamic viscosity [Pa·s]
    rho_test = 1.         # density [kg/m³]
    T_fluid = 400.0       # initial fluid temperature [K]
    T_wall = 300.0        # wall temperature (both walls) [K]

    # No flow conditions
    jx_test = 0.0
    jy_test = 0.0
    U_test = 0.0
    V_test = 0.0

    # Energy from temperature (no kinetic energy)
    E_test = T_fluid * rho_test * cv_test

    print(f"\nInput parameters:")
    print(f"  h       = {h_test:.2e} m")
    print(f"  h_eff   = {h_eff_test:.2e}")
    print(f"  k       = {k_test:.4f} W/(m·K)")
    print(f"  cv      = {cv_test:.2f} J/(kg·K)")
    print(f"  eta     = {eta_test:.2e} Pa·s")
    print(f"  rho     = {rho_test:.4f} kg/m³")
    print(f"  T_fluid = {T_fluid:.2f} K")
    print(f"  T_wall  = {T_wall:.2f} K (both walls)")
    print(f"  E       = {E_test:.2f} J/m³")
    print(f"  jx, jy  = {jx_test}, {jy_test} (no flow)")
    print(f"  U, V    = {U_test}, {V_test} (no wall movement)")

    q_top_test = heatflux_top(h_test, h_eff_test, k_test, cv_test, eta_test,
                              rho_test, E_test, jx_test, jy_test,
                              U_test, V_test, T_wall, T_wall, None)
    q_bot_test = heatflux_bot(h_test, h_eff_test, k_test, cv_test, eta_test,
                              rho_test, E_test, jx_test, jy_test,
                              U_test, V_test, T_wall, T_wall, None)

    print(f"\nResults:")
    print(f"  q_top = {q_top_test:.6e} W/m³")
    print(f"  q_bot = {q_bot_test:.6e} W/m³")
    print(f"  q_sum = {q_top_test + q_bot_test:.6e} W/m³")
    print(f"  Ratio q_top/q_bot = {q_top_test / q_bot_test:.2f}")

    print(f"\nExpected behavior:")
    print(f"  With symmetric walls and no flow, q_top should equal q_bot")
    print(f"  Current ratio indicates asymmetry in the formulas")
