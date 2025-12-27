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
from mpi4py import MPI
import numpy as np
import sys
import os

hans_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, hans_dir)

from GaPFlow.problem import Problem
from GaPFlow.models.pressure import eos_pressure


CONFIG = """
options:
    output: /tmp/fem2d_sphere
    write_freq: 1
    silent: False

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: 64
    Ny: 64
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['D', 'N', 'N']
    yN: ['D', 'N', 'N']
    xE_D: 1.1853
    xW_D: 1.1853
    yS_D: 1.1853
    yN_D: 1.1853

geometry:
    type: inclined
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
    dynamic: True
    R_norm_tol: 1e-11
    pressure_stab_alpha: 500
    equations:
        term_list: ['R11x', 'R11y', 'R11Sx', 'R11Sy', 'R1Stabx', 'R1Staby', 'R1T', 'R21x', 'R21y', 'R24x', 'R24y', 'R2Tx', 'R2Ty']
"""


def set_spherical_topography(problem, hmin, hmax):
    """
    Set a spherical topography with minimum height at center, maximum at corners.

    h(x, y) = hmin + (hmax - hmin) * r^2 / r_max^2

    where r^2 = (x - x_c)^2 + (y - y_c)^2 and r_max^2 = (Lx/2)^2 + (Ly/2)^2
    """
    Lx = problem.grid['Lx']
    Ly = problem.grid['Ly']

    # Get coordinates from topography
    xx = problem.topo.x
    yy = problem.topo.y

    # Center of domain
    x_c = Lx / 2
    y_c = Ly / 2

    # Maximum radius (corner distance from center)
    r_max_sq = (Lx / 2)**2 + (Ly / 2)**2

    # Compute distance from center
    r_sq = (xx - x_c)**2 + (yy - y_c)**2

    # Spherical height profile: minimum at center, maximum at corners
    h = hmin + (hmax - hmin) * r_sq / r_max_sq

    # Set the height field (gradients are computed automatically via the setter)
    problem.topo.h = h

    return h


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
    jx_local = q[nb:2*nb].reshape((solver.Ny_inner, solver.Nx_inner))
    jy_local = q[2*nb:3*nb].reshape((solver.Ny_inner, solver.Nx_inner))

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


def gather_global_height(problem):
    """Gather height field from all MPI ranks to rank 0."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    decomp = problem.decomp
    solver = problem.solver

    # Get local height field from solver's nodal fields (excludes ghost cells)
    # The nodal field 'h' has shape matching inner nodes
    h_nodal = solver.nodal_fields['h'].pg[0]

    # h_nodal is (Nx_inner, Ny_inner) for the local subdomain
    Nx_inner = solver.Nx_inner
    Ny_inner = solver.Ny_inner

    # Extract the inner part (should already be inner only)
    h_local = h_nodal[:Nx_inner, :Ny_inner]

    # Global dimensions
    Nx_global, Ny_global = decomp.nb_domain_grid_pts

    # Gather local Y sizes from all ranks
    all_ny = comm.allgather(Ny_inner)

    # Gather subdomain Y locations
    loc_y = decomp.subdomain_locations[1]
    all_loc_y = comm.allgather(loc_y)

    # Gather all local arrays to rank 0
    all_h = comm.gather(h_local, root=0)

    if rank == 0:
        # Assemble global array - h is stored as (Nx, Ny)
        h_global = np.zeros((Nx_global, Ny_global))

        for r in range(size):
            y_start = all_loc_y[r]
            y_end = y_start + all_ny[r]
            h_global[:, y_start:y_end] = all_h[r]

        return h_global
    else:
        return None


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

    # Set spherical topography BEFORE running
    hmin = problem.geo['hmin']
    hmax = problem.geo['hmax']
    set_spherical_topography(problem, hmin, hmax)

    if rank == 0:
        print(f"\nGrid: {problem.grid['Nx']}x{problem.grid['Ny']}")
        print(f"dt = {problem.numerics['dt']}, max_it = {problem.numerics['max_it']}")
        print(f"Dynamic mode: {problem.fem_solver['dynamic']}")
        print(f"Topography: Spherical (hmin={hmin:.2e} at center, hmax={hmax:.2e} at corners)")

    # Run simulation
    if rank == 0:
        print("\nRunning simulation...")
        print("-" * 70)

    problem.run()

    # Gather full solution from all ranks
    rho, jx, jy, Nx_global, Ny_global = gather_global_solution(problem)
    h_global = gather_global_height(problem)

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

        print("\n" + "=" * 70)
        print("Spherical topography - pressure buildup at center expected")
        print("=" * 70)

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
    x = np.linspace(dx/2, Nx * dx - dx/2, Nx)
    y = np.linspace(dy/2, Ny * dy - dy/2, Ny)
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

    # 2. Density field with flux vectors
    ax2 = axes[0, 1]
    c2 = ax2.contourf(X * 1000, Y * 1000, rho, levels=20, cmap='viridis')
    plt.colorbar(c2, ax=ax2, label=r'$\rho$ [kg/m³]')
    # Quiver plot for flux
    skip = max(1, Nx // 10)
    ax2.quiver(X[::4, ::skip] * 1000, Y[::4, ::skip] * 1000,
               jx[::4, ::skip], jy[::4, ::skip],
               color='white', alpha=0.8, scale=0.02)
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    ax2.set_title(r'Density $\rho$ with flux vectors $\mathbf{j}$')
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
