from GaPFlow.models.continuum_wall_stress import TTSlip, NewtonParabolicFlowFluidGap, SlipLengthSlip
import jax.numpy as jnp
import numpy as np


def test_base_gap_wall_stress_couette():
    mu0 = 1
    gap = NewtonParabolicFlowFluidGap(mu0)

    tbot, ttop = gap.wall_stress(1, 0, 0, 1, 1, -1)

    # The wall stress is the stress in the fluid at the wall

    # Here we are applying a negative shear rate to the fluid, so we expect the stress to be negative too
    print(tbot, ttop)
    assert tbot < 0
    assert ttop < 0

    assert abs(tbot - tbot) / abs(tbot) < 1e-3
    # Lets say the positive velocity is to the right

    # tbot is negative because the fluid is pulling the bottom wall to the left
    # ttop is positive because the fluid is pushing the top wall to the right

    # The motion of the walls is
    # <- <- <-
    #
    # ->  ->  ->

    # Which is a negative shear rate if the z direction points to the top

    # The stress from the wall on the fluid is eactly in that direction,
    # and so the stress in the fluid

    #
    # --------------------------------
    # \
    #  \
    #   \
    #    \
    # -------------

def test_base_gap_wall_stress_poiseuille():
    mu0 = 1
    gap = NewtonParabolicFlowFluidGap(mu0)

    tbot, ttop = gap.wall_stress(1, 0, 1, 1, 0, 0)
    assert ttop < 0
    assert tbot > 0

    # assert the stresses are the negatives of each other
    assert abs(ttop + tbot)  /  abs(ttop) < 1e-3
    print(tbot, ttop)

    # --------------
    # \
    #  \
    #   \
    #   /
    #  /
    # /
    # -------------



def test_newton_wall_stress_2D_x_flow_matches_1D():
    """wall_stress_2D with pure x-flow must reproduce wall_stress x-components; y-components must be zero."""
    mu0 = 0.37e-3
    gap = NewtonParabolicFlowFluidGap(mu0)

    cases = [
        # (rho, theta, j, h, U_bot, U_top)
        (700.0, 0.0, 0.0,   1e-6,  1.0, -1.0),   # pure Couette
        (700.0, 0.0, 1e-4,  1e-6,  0.0,  0.0),   # pure Poiseuille
        (700.0, 0.3, 5e-5,  2e-6,  0.5, -0.5),   # mixed with cavitation
    ]
    for rho, theta, j, h, U_bot, U_top in cases:
        tau_bot_1d, tau_top_1d = gap.wall_stress(rho, theta, j, h, U_bot, U_top)
        tau_bot_x, tau_top_x, tau_bot_y, tau_top_y = gap.wall_stress_2D(
            rho, theta, j, 0.0, h, 0.0,
            U_bot, U_top, 0.0, 0.0
        )
        assert abs(float(tau_bot_x) - float(tau_bot_1d)) / (abs(float(tau_bot_1d)) + 1e-30) < 1e-10, \
            f"tau_bot_x mismatch: 2D={tau_bot_x:.4g}, 1D={tau_bot_1d:.4g}"
        assert abs(float(tau_top_x) - float(tau_top_1d)) / (abs(float(tau_top_1d)) + 1e-30) < 1e-10, \
            f"tau_top_x mismatch: 2D={tau_top_x:.4g}, 1D={tau_top_1d:.4g}"
        assert abs(float(tau_bot_y)) < 1e-12, f"tau_bot_y should be zero, got {tau_bot_y:.4g}"
        assert abs(float(tau_top_y)) < 1e-12, f"tau_top_y should be zero, got {tau_top_y:.4g}"


def test_newton_wall_stress_2D_y_flow_matches_1D():
    """wall_stress_2D with pure y-flow: y-components must match wall_stress; x-components must be zero."""
    mu0 = 0.37e-3
    gap = NewtonParabolicFlowFluidGap(mu0)

    cases = [
        (700.0, 0.0, 0.0,   1e-6,  1.0, -1.0),
        (700.0, 0.0, 1e-4,  1e-6,  0.0,  0.0),
        (700.0, 0.3, 5e-5,  2e-6,  0.5, -0.5),
    ]
    for rho, theta, j, h, V_bot, V_top in cases:
        tau_bot_1d, tau_top_1d = gap.wall_stress(rho, theta, j, h, V_bot, V_top)
        tau_bot_x, tau_top_x, tau_bot_y, tau_top_y = gap.wall_stress_2D(
            rho, theta, 0.0, j, h, 0.0,  0.0, 0.0, V_bot, V_top
        )
        assert abs(float(tau_bot_y) - float(tau_bot_1d)) / (abs(float(tau_bot_1d)) + 1e-30) < 1e-10, \
            f"tau_bot_y mismatch: 2D={tau_bot_y:.4g}, 1D={tau_bot_1d:.4g}"
        assert abs(float(tau_top_y) - float(tau_top_1d)) / (abs(float(tau_top_1d)) + 1e-30) < 1e-10, \
            f"tau_top_y mismatch: 2D={tau_top_y:.4g}, 1D={tau_top_1d:.4g}"
        assert abs(float(tau_bot_x)) < 1e-12, f"tau_bot_x should be zero, got {tau_bot_x:.4g}"
        assert abs(float(tau_top_x)) < 1e-12, f"tau_top_x should be zero, got {tau_top_x:.4g}"

