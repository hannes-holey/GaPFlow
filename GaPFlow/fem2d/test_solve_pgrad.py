#!/usr/bin/env python3
"""
2D FEM solve test using PETSc via problem.run().

Run serial:
    python test_solve.py

Run with MPI:
    mpirun -n 2 python test_solve.py
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


CONFIG = """
options:
    output: /tmp/fem2d_solve
    write_freq: 1
    silent: False

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: 128
    Ny: 128
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 1.0
    xW_D: 1.1

geometry:
    type: inclined
    hmax: 1e-5
    hmin: 1e-5
    U: 0.0
    V: 0.0

numerics:
    solver: fem
    dt: 10.
    tol: 1e-6
    max_it: 10

properties:
    EOS: PL
    rho0: 1.0
    shear: 1e-3
    bulk: 0.
    P0: 101325
    alpha: 0.

fem_solver:
    type: newton_alpha
    dynamic: True
    R_norm_tol: 1e-08
    pressure_stab_alpha: 100.0
    equations:
        term_list: ['R11x', 'R11y', 'R1T', 'R21x', 'R21y', 'R24x', 'R24y', 'R2Tx', 'R2Ty',
                    'R1Sx', 'R1Sy', 'R1Stabx', 'R1Staby']
"""


def gather_global_solution(problem):
    """
    Gather solution from all MPI ranks to rank 0.

    Domain decomposition is along Y-axis only, so each rank has:
    - Full X dimension (Nx_inner = Nx_global)
    - Partial Y dimension (local Ny_inner)

    Returns (rho, jx, jy) as full global arrays on rank 0, None on other ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    solver = problem.solver
    decomp = problem.decomp

    # Get local solution
    nb = solver.nb_inner_pts
    q = solver.get_q_nodal()

    # Local arrays (Ny_local, Nx) - note: reshape is (Ny, Nx) for row-major
    rho_local = q[0:nb].reshape((solver.Ny_inner, solver.Nx_inner))
    jx_local = q[nb:2 * nb].reshape((solver.Ny_inner, solver.Nx_inner))
    jy_local = q[2 * nb:3 * nb].reshape((solver.Ny_inner, solver.Nx_inner))

    # Global dimensions
    Nx_global, Ny_global = decomp.nb_domain_grid_pts

    # Gather local Y sizes from all ranks
    local_ny = solver.Ny_inner
    all_ny = comm.allgather(local_ny)

    # Gather subdomain Y locations
    loc_y = decomp.subdomain_locations[1]
    all_loc_y = comm.allgather(loc_y)

    # Gather all local arrays to rank 0
    all_rho = comm.gather(rho_local, root=0)
    all_jx = comm.gather(jx_local, root=0)
    all_jy = comm.gather(jy_local, root=0)

    if rank == 0:
        # Assemble global arrays
        rho_global = np.zeros((Ny_global, Nx_global))
        jx_global = np.zeros((Ny_global, Nx_global))
        jy_global = np.zeros((Ny_global, Nx_global))

        for r in range(size):
            y_start = all_loc_y[r]
            y_end = y_start + all_ny[r]
            rho_global[y_start:y_end, :] = all_rho[r]
            jx_global[y_start:y_end, :] = all_jx[r]
            jy_global[y_start:y_end, :] = all_jy[r]

        return rho_global, jx_global, jy_global, Nx_global, Ny_global
    else:
        return None, None, None, None, None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print("2D FEM Solve Test with PETSc")
        print("=" * 70)
        print(f"Running on {size} MPI rank(s)")

    # Create problem
    problem = Problem.from_string(CONFIG)

    if rank == 0:
        print(f"\nGrid: {problem.grid['Nx']}x{problem.grid['Ny']}{problem.decomp.subdomain_info}")
        print(f"dt = {problem.numerics['dt']}, max_it = {problem.numerics['max_it']}")
        print(f"Dynamic mode: {problem.fem_solver['dynamic']}")

    # Run simulation
    if rank == 0:
        print("\nRunning simulation...")
        print("-" * 70)

    problem.run()

    # Gather full solution from all ranks
    rho, jx, jy, Nx_global, Ny_global = gather_global_solution(problem)

    # Analyze and plot on rank 0
    if rank == 0:
        print("-" * 70)
        print("\nResults (global domain):")

        solver = problem.solver

        # Expected values
        P0 = problem.prop['P0']
        mu = problem.prop['shear']
        h = problem.geo['hmax']
        Lx = problem.grid['Lx']

        dp_dx_bc = P0 * (1.0 - 1.1) / Lx
        jx_expected = -dp_dx_bc * rho.mean() * h**2 / (12 * mu)

        jx_mean = jx.mean()
        jx_error = (jx_mean - jx_expected) / jx_expected * 100

        print(f"  Global grid: {Nx_global}x{Ny_global}")
        print(f"  rho: [{rho.min():.6f}, {rho.max():.6f}]")
        print(f"  jx:  [{jx.min():.6e}, {jx.max():.6e}]")
        print(f"  jy:  [{jy.min():.6e}, {jy.max():.6e}]")
        print(f"\n  Poiseuille estimate jx: {jx_expected:.6e}")
        print(f"  Computed mean jx: {jx_mean:.6e}")
        print(f"  Difference: {jx_error:+.2f}%")
        print("  (Note: Poiseuille formula is approximate for this geometry)")

        print("\n" + "=" * 70)
        print("Pressure-driven flow test completed")
        print("=" * 70)

        # Generate plots with full domain
        print("\nGenerating plots...")
        plot_solution(solver, rho, jx, jy, problem, jx_expected, Nx_global, Ny_global)


def plot_solution(solver, rho, jx, jy, problem, jx_expected, Nx_global=None, Ny_global=None):
    """Plot the 2D FEM solution."""
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

    # Compute pressure from density (EOS: PL means p = P0 * rho)
    P0 = problem.prop.get('P0', 101325)
    p = P0 * rho

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Density field with flux vectors
    ax1 = axes[0, 0]
    c1 = ax1.contourf(X * 1000, Y * 1000, rho, levels=20, cmap='viridis')
    plt.colorbar(c1, ax=ax1, label=r'$\rho$ [kg/m³]')
    # Quiver plot for flux
    skip = max(1, Nx // 10)
    ax1.quiver(X[::2, ::skip] * 1000, Y[::2, ::skip] * 1000,
               jx[::2, ::skip], jy[::2, ::skip],
               color='white', alpha=0.8, scale=0.015)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    ax1.set_title(r'Density $\rho$ with flux vectors $\mathbf{j}$')
    ax1.set_aspect('equal')

    # 2. Pressure field
    ax2 = axes[0, 1]
    c2 = ax2.contourf(X * 1000, Y * 1000, p / 1000, levels=20, cmap='coolwarm')
    plt.colorbar(c2, ax=ax2, label='p [kPa]')
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    ax2.set_title(r'Pressure field $p = P_0 \rho$')
    ax2.set_aspect('equal')

    # 3. Mass flux jx field
    ax3 = axes[1, 0]
    c3 = ax3.contourf(X * 1000, Y * 1000, jx * 1000, levels=20, cmap='plasma')
    plt.colorbar(c3, ax=ax3, label=r'$j_x \times 10^3$')
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('y [mm]')
    ax3.set_title(r'Mass flux $j_x$')
    ax3.set_aspect('equal')

    # 4. Profiles with expected value comparison
    ax4 = axes[1, 1]
    mid_y = rho.shape[0] // 2
    x_line = X[mid_y, :] * 1000
    p_line = p[mid_y, :] / 1000  # kPa

    # Pressure on left axis
    color_p = 'tab:red'
    ax4.set_xlabel('x [mm]')
    ax4.set_ylabel('p [kPa]', color=color_p)
    ax4.plot(x_line, p_line, color=color_p, linewidth=2, label='p(x)')
    ax4.tick_params(axis='y', labelcolor=color_p)

    # jx on right axis
    ax4_twin = ax4.twinx()
    color_j = 'tab:blue'
    ax4_twin.set_ylabel(r'$j_x \times 10^3$ [kg/(m²·s)]', color=color_j)
    ax4_twin.axhline(y=jx_expected * 1000, color='green', linestyle=':', linewidth=2,
                     label=f'Expected jx = {jx_expected * 1000:.3f}')
    ax4_twin.plot(x_line, jx[mid_y, :] * 1000, color=color_j, linewidth=2,
                  label=f'Computed jx (mean = {jx.mean() * 1000:.3f})')
    ax4_twin.tick_params(axis='y', labelcolor=color_j)

    ax4.set_title('Centerline profiles')
    ax4.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)

    # Add boundary condition annotations
    fig.text(0.5, 0.01,
             r'BCs: $\rho_W = 1.1$, $\rho_E = 1.0$ (Dirichlet), '
             r'$\partial j/\partial n = 0$ at E/W (Neumann), Periodic in Y',
             ha='center', fontsize=9, style='italic')

    plt.suptitle('2D FEM Solution (PETSc): Pressure-Driven Flow', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    main()
