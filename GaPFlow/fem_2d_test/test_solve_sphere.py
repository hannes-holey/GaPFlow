#!/usr/bin/env python3
"""
2D FEM solve test with spherical topography using PETSc via problem.run().

The topography has minimum height at the center and maximum height at the corners,
like a ball pressed into a flat surface (convex sphere).

Run serial:
    python test_solve_sphere.py

Run with MPI:
    mpirun -n 2 python test_solve_sphere.py
"""
import os
import resource
import sys

# Limit memory to 12GB to prevent OOM crashes in WSL
_MEM_LIMIT_GB = 12
resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_GB * 1024**3, _MEM_LIMIT_GB * 1024**3))

hans_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, hans_dir)

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402
from GaPFlow.problem import Problem  # noqa: E402
from GaPFlow.models.pressure import eos_pressure  # noqa: E402


CONFIG = """
options:
    output: data/fem2d_sphere
    write_freq: 1
    silent: False

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: 128
    Ny: 128
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
    dt: 10.
    tol: 0.
    max_it: 10

properties:
    EOS: PL
    shear: 1.846e-5
    bulk: 0.
    P0: 101325
    rho0: 1.1853
    alpha: 0.

fem_solver:
    type: newton_alpha
    linear_solver: direct
    dynamic: True
    R_norm_tol: 1e-11
    pressure_stab_alpha: 500
    equations:
        term_list: ['R11x', 'R11y', 'R11Sx', 'R11Sy', 'R1Stabx', 'R1Staby', 'R1T',
                    'R21x', 'R21y', 'R24x', 'R24y', 'R2Tx', 'R2Ty']
"""


def gather_global_solution(problem):
    """Gather solution fields from all ranks using DomainDecomposition."""
    decomp = problem.decomp
    rho = decomp.gather_global(problem.q[0])
    jx = decomp.gather_global(problem.q[1])
    jy = decomp.gather_global(problem.q[2])
    # Transpose from (Nx, Ny) to (Ny, Nx) for plotting compatibility
    if decomp.rank == 0:
        return rho.T, jx.T, jy.T
    return None, None, None


def gather_global_height(problem):
    """Gather height field from all ranks using DomainDecomposition."""
    return problem.decomp.gather_global(problem.topo.h)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print("2D FEM Solve Test: Spherical Topography")
        print("=" * 70)
        print(f"Running on {size} MPI rank(s)")

    # Create problem
    problem = Problem.from_string(CONFIG)

    hmin = problem.geo['hmin']
    hmax = problem.geo['hmax']

    if rank == 0:
        print(f"\nGrid: {problem.grid['Nx']}x{problem.grid['Ny']}{problem.decomp.subdomain_info}")
        print(f"dt = {problem.numerics['dt']}, max_it = {problem.numerics['max_it']}")
        print(f"Dynamic mode: {problem.fem_solver['dynamic']}")
        print(f"Topography: parabolic_2d (hmin={hmin:.2e} at center, hmax={hmax:.2e} at corners)")

    # Run simulation
    if rank == 0:
        print("\nRunning simulation...")
        print("-" * 70)

    problem.run()

    # Gather full solution from all ranks
    rho, jx, jy = gather_global_solution(problem)
    h_global = gather_global_height(problem)
    Nx_global, Ny_global = problem.decomp.nb_domain_grid_pts

    # Analyze and plot on rank 0
    if rank == 0:
        print("-" * 70)
        print("\nResults (global domain):")

        solver = problem.solver

        print(f"  Global grid: {Nx_global}x{Ny_global}")
        print(f"  rho: [{rho.min():.6f}, {rho.max():.6f}]")
        print(f"  jx:  [{jx.min():.6e}, {jx.max():.6e}]")
        print(f"  jy:  [{jy.min():.6e}, {jy.max():.6e}]")
        print(f"  h:   [{h_global.min():.6e}, {h_global.max():.6e}]")

        solver.print_timer_summary()

        # Generate plots with full domain
        print("\nGenerating plots...")
        plot_solution(solver, rho, jx, jy, h_global, problem, Nx_global, Ny_global)


def plot_solution(solver, rho, jx, jy, h, problem, Nx_global=None, Ny_global=None):
    """Plot the 2D FEM solution for spherical topography."""
    import matplotlib
    matplotlib.rcParams['figure.constrained_layout.use'] = False
    import matplotlib.pyplot as plt

    # Use global dimensions if provided, else use local
    Nx = Nx_global if Nx_global is not None else solver.Nx_inner
    Ny = Ny_global if Ny_global is not None else solver.Ny_inner
    dx, dy = solver.dx, solver.dy

    # Grid coordinates (cell centers)
    x = np.linspace(dx / 2, Nx * dx - dx / 2, Nx)
    y = np.linspace(dy / 2, Ny * dy - dy / 2, Ny)
    X, Y = np.meshgrid(x, y)

    # Compute pressure from density using the actual EOS from problem
    p = np.asarray(eos_pressure(rho, problem.prop))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # 1. Height field
    ax1 = axes[0, 0]
    # h is stored as (Nx, Ny), transpose for plotting
    c1 = ax1.contourf(X * 1000, Y * 1000, h.T * 1e6, levels=20, cmap='terrain')
    plt.colorbar(c1, ax=ax1, label=r'$h$ [$\mu$m]')
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    ax1.set_title(r'Gap height $h$ (spherical)')
    ax1.set_aspect('equal')

    # 2. Density field with flux deviation vectors (relative to mean Couette flow)
    ax2 = axes[0, 1]
    c2 = ax2.contourf(X * 1000, Y * 1000, rho, levels=20, cmap='viridis')
    plt.colorbar(c2, ax=ax2, label=r'$\rho$ [kg/m³]')
    # Quiver plot for flux deviation - subtract mean advective flux U/2 * rho
    U = problem.geo['U']
    jx_dev = jx - (U / 2) * rho
    skip = max(1, Nx // 12)
    jx_skip, jy_skip = jx_dev[::skip, ::skip], jy[::skip, ::skip]
    j_mag = np.sqrt(jx_skip**2 + jy_skip**2)
    j_mag[j_mag == 0] = 1  # Avoid division by zero
    jx_norm, jy_norm = jx_skip / j_mag, jy_skip / j_mag
    ax2.quiver(X[::skip, ::skip] * 1000, Y[::skip, ::skip] * 1000,
               jx_norm, jy_norm,
               color='white', alpha=0.9, scale=15, width=0.004, headwidth=3, headlength=4)
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    ax2.set_title(r'Density $\rho$ with flux deviation $\mathbf{j} - \frac{U}{2}\rho$')
    ax2.set_aspect('equal')

    # 3. Pressure field
    ax3 = axes[0, 2]
    c3 = ax3.contourf(X * 1000, Y * 1000, p / 1000, levels=20, cmap='coolwarm')
    plt.colorbar(c3, ax=ax3, label='p [kPa]')
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('y [mm]')
    ax3.set_title(f'Pressure field (EOS: {problem.prop["EOS"]})')
    ax3.set_aspect('equal')

    # 4. Mass flux jx field
    ax4 = axes[1, 0]
    c4 = ax4.contourf(X * 1000, Y * 1000, jx * 1000, levels=20, cmap='plasma')
    plt.colorbar(c4, ax=ax4, label=r'$j_x \times 10^3$')
    ax4.set_xlabel('x [mm]')
    ax4.set_ylabel('y [mm]')
    ax4.set_title(r'Mass flux $j_x$')
    ax4.set_aspect('equal')

    # 5. Mass flux jy field
    ax5 = axes[1, 1]
    c5 = ax5.contourf(X * 1000, Y * 1000, jy * 1000, levels=20, cmap='plasma')
    plt.colorbar(c5, ax=ax5, label=r'$j_y \times 10^3$')
    ax5.set_xlabel('x [mm]')
    ax5.set_ylabel('y [mm]')
    ax5.set_title(r'Mass flux $j_y$')
    ax5.set_aspect('equal')

    # 6. Centerline profiles
    ax6 = axes[1, 2]
    mid_y = rho.shape[0] // 2
    mid_x = rho.shape[1] // 2
    x_line = X[mid_y, :] * 1000
    y_line = Y[:, mid_x] * 1000
    p_x = p[mid_y, :] / 1000  # kPa along x at y=Ly/2
    p_y = p[:, mid_x] / 1000  # kPa along y at x=Lx/2

    ax6.plot(x_line, p_x, 'b-', linewidth=2, label=r'$p(x, L_y/2)$')
    ax6.plot(y_line, p_y, 'r--', linewidth=2, label=r'$p(L_x/2, y)$')
    ax6.set_xlabel('Position [mm]')
    ax6.set_ylabel('p [kPa]')
    ax6.set_title('Centerline pressure profiles')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add boundary condition annotations
    fig.text(0.5, 0.01,
             r'BCs: $\rho = 1.1853$ (Dirichlet all sides), '
             r'$\partial j/\partial n = 0$ (Neumann), U=50 m/s, V=0',
             ha='center', fontsize=9, style='italic')

    plt.suptitle('2D FEM Solution: Spherical Topography (hmin at center)', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    main()
