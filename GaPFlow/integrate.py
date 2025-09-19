import numpy as np


def predictor_corrector(q, h, p, tau, direction):

    FxH, FyH = hyperbolicFlux(q, p)
    FxD, FyD = diffusiveFlux(q, h, tau)

    Fx = FxH + FxD
    Fy = FyH + FyD

    flux_x = -direction * (np.roll(Fx, direction, axis=1) - Fx)
    flux_y = -direction * (np.roll(Fy, direction, axis=2) - Fy)

    return flux_x, flux_y


def source(q, h, stress, stress_lower, stress_upper):

    out = np.zeros_like(q)

    # origin bottom, U_top = 0, U_bottom = U
    out[0] = (-q[1] * h[1] - q[2] * h[2]) / h[0]

    out[1] = ((stress[0] - stress_upper[0]) * h[1] +  # noqa: W504
              (stress[2] - stress_upper[5]) * h[2] +  # noqa: W504
              stress_upper[4] - stress_lower[4]) / h[0]

    out[2] = ((stress[2] - stress_upper[5]) * h[1] +  # noqa: W504
              (stress[1] - stress_upper[1]) * h[2] +  # noqa: W504
              stress_upper[3] - stress_lower[3]) / h[0]

    return out


def hyperbolicFlux(q, p):

    Fx = np.zeros_like(q)
    Fy = np.zeros_like(q)

    # x
    Fx[0] = q[1]
    Fx[1] = p
    # y
    Fy[0] = q[2]
    Fy[2] = p

    return Fx, Fy


def diffusiveFlux(q, h, tau):

    Dx = np.zeros_like(q)
    Dy = np.zeros_like(q)

    Dx[1] = tau[0]
    Dx[2] = tau[2]

    Dy[1] = tau[2]
    Dy[2] = tau[1]

    return Dx, Dy
