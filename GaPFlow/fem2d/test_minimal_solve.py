#!/usr/bin/env python3
"""
Minimal 2D FEM solve test - Periodic Y with pressure gradient in X.

This configuration is well-posed:
- Periodic BCs in Y direction for jx, jy (breaks checkerboard null space)
- Dirichlet on rho at xE/xW boundaries (pressure gradient in X)
- Neumann on jx, jy at xE/xW boundaries

Run: python test_minimal_solve.py
"""
from mpi4py import MPI
import numpy as np
import sys
import os
import time

hans_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, hans_dir)

from GaPFlow.problem import Problem


class Timer:
    """Simple timer for benchmarking."""
    def __init__(self):
        self.times = {}
        self.counts = {}
        self._start_time = None
        self._current_name = None

    def start(self, name):
        self._current_name = name
        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is None:
            return
        elapsed = time.perf_counter() - self._start_time
        name = self._current_name
        if name not in self.times:
            self.times[name] = 0.0
            self.counts[name] = 0
        self.times[name] += elapsed
        self.counts[name] += 1
        self._start_time = None
        self._current_name = None
        return elapsed

    def report(self):
        print("\n" + "=" * 70)
        print("TIMING BENCHMARK")
        print("=" * 70)
        print(f"{'Operation':<35} {'Total [s]':>12} {'Count':>8} {'Avg [ms]':>12}")
        print("-" * 70)
        total = 0.0
        for name in self.times:
            t = self.times[name]
            n = self.counts[name]
            avg_ms = (t / n) * 1000 if n > 0 else 0
            print(f"{name:<35} {t:>12.4f} {n:>8d} {avg_ms:>12.4f}")
            total += t
        print("-" * 70)
        print(f"{'TOTAL':<35} {total:>12.4f}")
        print("=" * 70)


timer = Timer()

# Periodic Y with pressure gradient in X
# Using DYNAMIC mode with time derivative terms for well-posed system
CONFIG = """
options:
    output: /tmp/fem2d_minimal
    write_freq: 1000
    silent: True

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: 100
    Ny: 16
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
    max_it: 100

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
    equations:
        term_list: ['R11x', 'R11y', 'R1T', 'R21x', 'R21y', 'R24x', 'R24y', 'R2Tx', 'R2Ty']
"""

# Number of time steps to reach steady state
N_STEPS = 10


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("=" * 70)
    print("Minimal 2D FEM Solve Test - Dynamic Mode with Time Stepping")
    print("=" * 70)
    print("\nSetup:")
    print("  - DYNAMIC mode with time derivative terms (R1T, R2Tx, R2Ty)")
    print("  - Periodic BCs in Y for all variables")
    print("  - Dirichlet rho: xW=1.1, xE=1.0 (pressure gradient in X)")
    print("  - Neumann for jx, jy at xE/xW")
    print("  - Uniform gap height h = 1e-5, no wall velocity")
    print(f"  - dt = 0.1 s, {N_STEPS} time steps")

    # Create problem
    timer.start("Problem.from_string")
    problem = Problem.from_string(CONFIG)
    timer.stop()
    solver = problem.solver

    # Initialize solver
    print("\n1. Initializing solver...")
    timer.start("solver.pre_run")
    solver.pre_run()
    timer.stop()

    print(f"   Grid: {solver.Nx_inner}x{solver.Ny_inner} inner points")
    print(f"   dx = {solver.dx:.6f}, dy = {solver.dy:.6f}")
    print(f"   nb_inner_pts = {solver.nb_inner_pts}")
    print(f"   Variables: {solver.variables}")
    print(f"   Terms: {[t.name for t in solver.terms]}")
    print(f"   Dynamic mode: {solver.dynamic}")

    nb = solver.nb_inner_pts

    # Apply boundary conditions and initialize
    print("\n2. Applying boundary conditions...")
    timer.start("communicate_ghost_buffers")
    problem.decomp.communicate_ghost_buffers(problem)
    timer.stop()

    timer.start("update_quad")
    solver.update_quad()
    timer.stop()
    solver.update_prev_quad()  # Set previous state for time derivative

    q = solver.get_q_nodal().copy()
    print(f"   Initial q shape: {q.shape}")
    print(f"   Initial rho: [{q[0:nb].min():.6f}, {q[0:nb].max():.6f}]")
    print(f"   Initial jx:  [{q[nb:2*nb].min():.6e}, {q[nb:2*nb].max():.6e}]")

    # Check matrix rank (with time derivatives, should be full rank)
    print("\n3. Matrix diagnostics (with time derivative terms)...")
    timer.start("solver_step_fun")
    M, R = solver.solver_step_fun(q)
    timer.stop()
    rank_M = np.linalg.matrix_rank(M)
    print(f"   Matrix rank: {rank_M} / {M.shape[0]}")
    print(f"   Initial |R|: {np.linalg.norm(R):.4e}")

    if rank_M < M.shape[0]:
        print(f"   WARNING: Matrix is rank deficient!")
        return
    print(f"   Matrix is FULL RANK - system is well-posed!")

    # Time stepping loop
    print(f"\n4. Time stepping ({N_STEPS} steps)...")
    print("=" * 70)

    tol = 1e-10
    dt = problem.numerics['dt']

    timer.start("Time stepping (total)")
    for step in range(N_STEPS):
        # Store previous state for time derivative
        solver.update_prev_quad()

        # Newton iteration within time step
        for it in range(30):
            timer.start("solver_step_fun")
            M, R = solver.solver_step_fun(q)
            timer.stop()
            R_norm = np.linalg.norm(R)

            if R_norm < tol:
                break

            timer.start("np.linalg.solve")
            dq = np.linalg.solve(M, -R)
            timer.stop()

            q = q + dq

            timer.start("set_q_nodal")
            solver.set_q_nodal(q)
            timer.stop()
            timer.start("communicate_ghost_buffers")
            problem.decomp.communicate_ghost_buffers(problem)
            timer.stop()
            timer.start("update_quad")
            solver.update_quad()
            timer.stop()

        # Report progress
        jx_mean = q[nb:2*nb].mean()
        print(f"   Step {step+1:3d}: t = {(step+1)*dt:.2e} s, |R| = {R_norm:.2e}, "
              f"Newton its = {it+1}, jx_mean = {jx_mean:.6e}")

    timer.stop()  # Time stepping (total)
    print("=" * 70)

    # Final solution
    print("\n5. Final solution...")
    rho_final = q[0:nb].reshape((solver.Ny_inner, solver.Nx_inner))
    jx_final = q[nb:2*nb].reshape((solver.Ny_inner, solver.Nx_inner))
    jy_final = q[2*nb:3*nb].reshape((solver.Ny_inner, solver.Nx_inner))

    print(f"   rho: [{rho_final.min():.6f}, {rho_final.max():.6f}]")
    print(f"   jx:  [{jx_final.min():.6e}, {jx_final.max():.6e}]")
    print(f"   jy:  [{jy_final.min():.6e}, {jy_final.max():.6e}]")

    # Physical analysis with CORRECT expected value
    print("\n6. Physical analysis...")

    props = problem.prop
    geom = problem.geo

    P0 = props.get('P0', 101325)
    mu = props.get('shear', 1e-3)
    h = geom.get('hmax', 1e-5)
    Lx = problem.grid['Lx']

    # Pressure gradient from boundary conditions
    drho_dx_bc = (1.0 - 1.1) / Lx  # = -1.0
    dp_dx_bc = P0 * drho_dx_bc      # = -101325 Pa/m

    # Computed pressure gradient
    drho_dx = (rho_final[:, -1] - rho_final[:, 0]).mean() / (solver.dx * (solver.Nx_inner - 1))
    dp_dx = P0 * drho_dx

    # CORRECT expected jx from wall stress model:
    # At steady state: dp/dx = -12*eta*jx/(rho*h²)
    # => jx = -dp/dx * rho * h² / (12*eta)
    rho_avg = rho_final.mean()
    jx_expected = -dp_dx_bc * rho_avg * h**2 / (12 * mu)

    # Note: Simple Poiseuille would give jx = -dp/dx * rho * h² / mu (wrong for this model)
    jx_poiseuille = -dp_dx_bc * rho_avg * h**2 / mu

    jx_mean = jx_final.mean()
    jx_error = (jx_mean - jx_expected) / jx_expected * 100

    print(f"\n   Boundary conditions:")
    print(f"     rho_W = 1.1, rho_E = 1.0")
    print(f"     dp/dx (from BCs) = {dp_dx_bc:.0f} Pa/m")

    print(f"\n   Computed gradients:")
    print(f"     drho/dx = {drho_dx:.4f} (expected: {drho_dx_bc:.4f})")
    print(f"     dp/dx   = {dp_dx:.0f} Pa/m")

    print(f"\n   Mass flux jx:")
    print(f"     Expected (wall stress model): jx = {jx_expected:.6e} kg/(m²·s)")
    print(f"     Computed:                     jx = {jx_mean:.6e} kg/(m²·s)")
    print(f"     Error: {jx_error:+.2f}%")

    print(f"\n   Note: Simple Poiseuille would give jx = {jx_poiseuille:.6e}")
    print(f"         (12x larger - WRONG for this wall stress model)")

    print(f"\n   jy (should be ~0): mean = {jy_final.mean():.2e}, max = {np.abs(jy_final).max():.2e}")

    # Pass/fail check
    print("\n" + "=" * 70)
    if abs(jx_error) < 5:
        print(f"Test PASSED - Solution within 5% of expected value!")
    else:
        print(f"Test FAILED - Solution error > 5%")
    print("=" * 70)

    # Print timing report
    timer.report()

    # Generate plots
    print("\n7. Generating plots...")
    plot_solution(solver, rho_final, jx_final, jy_final, problem, jx_expected)


def plot_solution(solver, rho, jx, jy, problem, jx_expected):
    """Plot the 2D FEM solution."""
    import matplotlib.pyplot as plt

    Nx, Ny = solver.Nx_inner, solver.Ny_inner
    dx, dy = solver.dx, solver.dy

    # Grid coordinates (cell centers)
    x = np.linspace(dx/2, Nx * dx - dx/2, Nx)
    y = np.linspace(dy/2, Ny * dy - dy/2, Ny)
    X, Y = np.meshgrid(x, y)

    # Compute velocity from flux: u = jx/rho, v = jy/rho
    u = jx / rho
    v = jy / rho

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
                     label=f'Expected jx = {jx_expected*1000:.3f}')
    ax4_twin.plot(x_line, jx[mid_y, :] * 1000, color=color_j, linewidth=2,
                  label=f'Computed jx (mean = {jx.mean()*1000:.3f})')
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

    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.06, hspace=0.28, wspace=0.30)
    plt.suptitle('2D FEM Solution: Dynamic Mode - Pressure-Driven Flow', fontsize=12)

    plt.show()


if __name__ == "__main__":
    main()
