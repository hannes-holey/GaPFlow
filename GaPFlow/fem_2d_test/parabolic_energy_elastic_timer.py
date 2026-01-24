#!/usr/bin/env python3
"""
Test script for FEM 2D solver with elastic deformation and energy equation.

Runs a parabolic slider use case and outputs detailed timer summary
to profile elastic deformation performance.

Usage:
    mpirun -n 4 python parabolic_energy_elastic_timer.py [--steps N]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from GaPFlow.problem import Problem  # noqa: E402
from GaPFlow.viz.plotting import _plot_sol_from_field_2d  # noqa: E402


# YAML configuration for parabolic slider + elastic deformation + energy
CONFIG_YAML = """
options:
    output: fem_2d_test/output_elastic_energy
    write_freq: 1
    silent: False

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: 100
    Ny: 100
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['D', 'N', 'N']
    yN: ['D', 'N', 'N']
    xE_D: 1.1853
    xW_D: 1.1853
    yS_D: 1.1853
    yN_D: 1.1853

geometry:
    type: parabolic_2d
    hmax: 6.6e-5
    hmin: 1.0e-5
    U: 50.
    V: 0.

numerics:
    solver: fem
    dt: 1e-04
    tol: 1e-10
    max_it: 20

properties:
    EOS: PL
    shear: 1.846e-5
    bulk: 0.
    P0: 101325
    rho0: 1.1853
    alpha: 0.
    elastic:
        E: 1e07
        v: 0.3
        alpha_underrelax: 0.1
        reference_point: [0, 50]

fem_solver:
    type: newton_alpha
    alpha: 1.0
    linear_solver: iterative
    max_iter: 30
    R_norm_tol: 1e-09
    pressure_stab_alpha: 4500
    momentum_stab_alpha: 25000
    energy_stab_alpha: 500000000
    equations:
        energy: True

energy_spec:
    T0: 300.
    T_wall: 300.
    bc_xW: 'D'
    bc_xE: 'N'
    bc_yS: 'N'
    bc_yN: 'N'
    T_bc_xW: 300.
"""


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(
        description="Run FEM 2D elastic+energy test with timer profiling"
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override max_it (number of timesteps)"
    )
    args = parser.parse_args()

    # Modify config if steps override provided
    config = CONFIG_YAML
    if args.steps is not None:
        config = config.replace("max_it: 20", f"max_it: {args.steps}")

    if rank == 0:
        print("=" * 70)
        print("FEM 2D Timer Test: Parabolic Slider + Elastic Deformation + Energy")
        print("=" * 70)

    # Create problem from YAML string
    problem = Problem.from_string(config)

    # Run simulation
    problem.run()

    # Print timer summary
    problem.solver.print_timer_summary()

    # Gather fields and plot on rank 0
    decomp = problem.decomp
    g = decomp.gather_global

    # Gather solution fields
    rho = g(problem.q[0])
    jx = g(problem.q[1])
    jy = g(problem.q[2])
    pressure = g(problem.pressure.pressure)
    tau_xz_bot = g(problem.wall_stress_xz.lower[4])
    tau_xz_top = g(problem.wall_stress_xz.upper[4])
    tau_yz_bot = g(problem.wall_stress_yz.lower[3])
    tau_yz_top = g(problem.wall_stress_yz.upper[3])

    if rank == 0:
        # Stack into q-like array with ghost cell padding for plotting
        Nx, Ny = rho.shape
        q_global = np.zeros((3, Nx + 2, Ny + 2))
        q_global[0, 1:-1, 1:-1] = rho
        q_global[1, 1:-1, 1:-1] = jx
        q_global[2, 1:-1, 1:-1] = jy

        # Pad other fields similarly
        def pad(f):
            out = np.zeros((Nx + 2, Ny + 2))
            out[1:-1, 1:-1] = f
            return out

        _plot_sol_from_field_2d(
            q_global, pad(pressure),
            pad(tau_xz_bot), pad(tau_xz_top),
            pad(tau_yz_bot), pad(tau_yz_top))
        plt.show()

        problem.solver.plot_residual_history()
        plt.show()


if __name__ == "__main__":
    main()
