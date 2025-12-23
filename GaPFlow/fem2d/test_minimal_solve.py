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
    dt: 1e-8
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
    dynamic: False
    equations:
        term_list: ['R11x', 'R11y', 'R21x', 'R21y', 'R24x', 'R24y']
"""


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("=" * 70)
    print("Minimal 2D FEM Solve Test - Periodic Y with Pressure Gradient in X")
    print("=" * 70)
    print("\nSetup:")
    print("  - Periodic BCs in Y for jx, jy (well-posed configuration)")
    print("  - Dirichlet rho: xW=1.1, xE=1.0 (pressure gradient in X)")
    print("  - Neumann for jx, jy at xE/xW")
    print("  - Uniform gap height h = 1e-5, no wall velocity")

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

    # Apply boundary conditions
    print("\n2. Applying boundary conditions...")
    timer.start("communicate_ghost_buffers")
    problem.decomp.communicate_ghost_buffers(problem)
    timer.stop()

    # Get initial solution
    print("\n3. Initial state...")
    timer.start("update_quad")
    solver.update_quad()
    timer.stop()
    q0 = solver.get_q_nodal()
    print(f"   q0 shape: {q0.shape}")

    nb = solver.nb_inner_pts
    rho_init = q0[0:nb]
    jx_init = q0[nb:2*nb]
    jy_init = q0[2*nb:3*nb]
    print(f"   Initial rho: [{rho_init.min():.6f}, {rho_init.max():.6f}]")
    print(f"   Initial jx:  [{jx_init.min():.6f}, {jx_init.max():.6f}]")
    print(f"   Initial jy:  [{jy_init.min():.6f}, {jy_init.max():.6f}]")

    # Check matrix rank
    print("\n4. Matrix diagnostics...")
    timer.start("get_M (Jacobian assembly)")
    M = solver.get_M()
    timer.stop()
    timer.start("get_R (Residual assembly)")
    R = solver.get_R()
    timer.stop()
    rank_M = np.linalg.matrix_rank(M)
    print(f"   Matrix rank: {rank_M} / {M.shape[0]}")
    print(f"   Initial |R|: {np.linalg.norm(R):.4e}")

    if rank_M < M.shape[0]:
        print(f"   WARNING: Matrix is rank deficient!")
        U_svd, s, Vh = np.linalg.svd(M)
        print(f"   Smallest singular values: {s[-5:]}")
        return

    print(f"   Condition number: {np.linalg.cond(M):.4e}")

    # Check individual term residuals
    print("\n5. Initial term residuals...")
    for term in solver.terms:
        R_term = solver.residual_vector_term(term)
        print(f"   {term.name}: |R| = {np.linalg.norm(R_term):.4e}")

    # Newton iteration with detailed diagnostics
    print("\n6. Newton iteration (with detailed diagnostics)...")
    print("=" * 70)

    q = solver.get_q_nodal().copy()
    tol = 1e-8
    max_iter = 50
    converged = False
    alpha = 1.0  # Full Newton step

    # Store history for analysis
    R_history = []
    dq_history = []

    timer.start("Newton iteration (total)")
    for it in range(max_iter):
        # Compute M and R at current state
        timer.start("solver_step_fun (M + R)")
        M, R = solver.solver_step_fun(q)
        timer.stop()
        R_norm = np.linalg.norm(R)
        R_history.append(R_norm)

        # Detailed output for first iterations
        if it < 10 or it % 10 == 0:
            print(f"\n--- Iteration {it} ---")
            print(f"   |R| = {R_norm:.6e}")

            # Residual by equation type
            R_mass = R[0:nb]
            R_momx = R[nb:2*nb]
            R_momy = R[2*nb:3*nb]
            print(f"   |R_mass|   = {np.linalg.norm(R_mass):.6e}")
            print(f"   |R_mom_x|  = {np.linalg.norm(R_momx):.6e}")
            print(f"   |R_mom_y|  = {np.linalg.norm(R_momy):.6e}")

            # Individual term contributions
            print(f"   Term contributions:")
            for term in solver.terms:
                R_term = solver.residual_vector_term(term)
                print(f"      {term.name}: |R| = {np.linalg.norm(R_term):.4e}")

        # Check convergence
        if R_norm < tol:
            converged = True
            print(f"\n   CONVERGED at iteration {it}!")
            break

        # Solve for update
        try:
            timer.start("np.linalg.solve")
            dq = np.linalg.solve(M, -R)
            timer.stop()
        except np.linalg.LinAlgError as e:
            timer.stop()
            print(f"\n   Linear solve FAILED at iteration {it}: {e}")
            break

        dq_norm = np.linalg.norm(dq)
        dq_history.append(dq_norm)

        # dq by variable
        dq_rho = dq[0:nb]
        dq_jx = dq[nb:2*nb]
        dq_jy = dq[2*nb:3*nb]

        if it < 10 or it % 10 == 0:
            print(f"\n   Solution update:")
            print(f"   |dq| = {dq_norm:.6e}")
            print(f"   |dq_rho| = {np.linalg.norm(dq_rho):.6e}, range: [{dq_rho.min():.4e}, {dq_rho.max():.4e}]")
            print(f"   |dq_jx|  = {np.linalg.norm(dq_jx):.6e}, range: [{dq_jx.min():.4e}, {dq_jx.max():.4e}]")
            print(f"   |dq_jy|  = {np.linalg.norm(dq_jy):.6e}, range: [{dq_jy.min():.4e}, {dq_jy.max():.4e}]")

            # Current state
            q_rho = q[0:nb]
            q_jx = q[nb:2*nb]
            q_jy = q[2*nb:3*nb]
            print(f"\n   Current state:")
            print(f"   rho: [{q_rho.min():.6f}, {q_rho.max():.6f}]")
            print(f"   jx:  [{q_jx.min():.6e}, {q_jx.max():.6e}]")
            print(f"   jy:  [{q_jy.min():.6e}, {q_jy.max():.6e}]")

        # Line search (simple backtracking if residual increases)
        alpha_used = alpha
        q_new = q + alpha_used * dq

        # Update state
        q = q_new

        # Update solver's internal state for next iteration
        timer.start("set_q_nodal")
        solver.set_q_nodal(q)
        timer.stop()
        timer.start("communicate_ghost_buffers")
        problem.decomp.communicate_ghost_buffers(problem)
        timer.stop()
        timer.start("update_quad")
        solver.update_quad()
        timer.stop()

    timer.stop()  # Newton iteration (total)
    print("\n" + "=" * 70)

    if converged:
        print(f"Newton iteration CONVERGED in {it} iterations")
        print(f"Final |R|: {R_norm:.6e}")
    else:
        print(f"Newton iteration did NOT converge in {max_iter} iterations")
        print(f"Final |R|: {R_norm:.6e}")

    # Show convergence history
    print("\n7. Convergence history...")
    print(f"   {'Iter':<6} {'|R|':<14} {'|dq|':<14} {'|R| ratio':<14}")
    print("   " + "-" * 48)
    for i in range(min(len(R_history), 20)):
        ratio = R_history[i] / R_history[i-1] if i > 0 else 0
        dq_val = dq_history[i] if i < len(dq_history) else 0
        print(f"   {i:<6} {R_history[i]:<14.6e} {dq_val:<14.6e} {ratio:<14.4f}")

    # Final solution analysis
    print("\n8. Final solution...")
    solver.set_q_nodal(q)
    problem.decomp.communicate_ghost_buffers(problem)

    q_final = solver.get_q_nodal()
    rho_final = q_final[0:nb].reshape((solver.Ny_inner, solver.Nx_inner))
    jx_final = q_final[nb:2*nb].reshape((solver.Ny_inner, solver.Nx_inner))
    jy_final = q_final[2*nb:3*nb].reshape((solver.Ny_inner, solver.Nx_inner))

    print(f"\n   Final rho (grid):")
    print(f"   {rho_final}")
    print(f"\n   Final jx (grid):")
    print(f"   {jx_final}")
    print(f"\n   Final jy (grid):")
    print(f"   {jy_final}")

    # Physical analysis
    print("\n9. Physical analysis...")

    # Access properties (Problem has 'prop' and 'geo' attributes)
    props = problem.prop
    geom = problem.geo

    # Pressure gradient check
    # With EOS: PL, p = P0 * rho, so dp/dx = P0 * drho/dx
    P0 = props.get('P0', 101325)
    drho_dx = (rho_final[:, -1] - rho_final[:, 0]).mean() / (solver.dx * (solver.Nx_inner - 1))
    dp_dx = P0 * drho_dx
    print(f"   Mean drho/dx: {drho_dx:.4f}")
    print(f"   Mean dp/dx:   {dp_dx:.2f} Pa/m")

    # Expected flow: with pressure gradient dp/dx and wall stress balance
    # dp/dx = tau_wall / h = mu/h^2 * jx/rho (for no wall motion)
    # => jx = rho * h^2 / mu * dp/dx
    mu = props.get('shear', 1e-3)
    h = geom.get('hmax', 1e-5)
    rho_avg = rho_final.mean()
    jx_expected = rho_avg * h**2 / mu * dp_dx
    print(f"\n   Expected jx from pressure-driven flow:")
    print(f"   jx_expected = rho * h^2 / mu * dp/dx = {jx_expected:.6e}")
    print(f"   Actual mean jx: {jx_final.mean():.6e}")

    # Mass conservation check
    print(f"\n   Mass conservation (∇·j should be ~0):")
    print(f"   Mean |jx|: {np.abs(jx_final).mean():.6e}")
    print(f"   Mean |jy|: {np.abs(jy_final).mean():.6e}")

    print("\n" + "=" * 70)
    if converged:
        print("Test PASSED - Newton iteration converged!")
    else:
        print("Test FAILED - Newton iteration did not converge")
    print("=" * 70)

    # Print timing report
    timer.report()

    # Generate plots
    print("\n10. Generating plots...")
    plot_solution(solver, rho_final, jx_final, jy_final, problem)


def plot_solution(solver, rho, jx, jy, problem):
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
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # 1. Density field with flux vectors
    ax1 = axes[0, 0]
    c1 = ax1.contourf(X * 1000, Y * 1000, rho, levels=20, cmap='viridis')
    plt.colorbar(c1, ax=ax1, label=r'$\rho$ [kg/m³]')
    # Quiver plot for flux
    skip = max(1, Nx // 12)
    ax1.quiver(X[::skip, ::skip] * 1000, Y[::skip, ::skip] * 1000,
               jx[::skip, ::skip], jy[::skip, ::skip],
               color='white', alpha=0.8, scale=0.02)
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

    # 3. Velocity magnitude with streamlines
    ax3 = axes[1, 0]
    speed = np.sqrt(u**2 + v**2)
    c3 = ax3.contourf(X * 1000, Y * 1000, speed * 1000, levels=20, cmap='plasma')
    plt.colorbar(c3, ax=ax3, label='|u| [mm/s]')
    # Streamlines
    ax3.streamplot(X * 1000, Y * 1000, u, v, color='white', linewidth=0.5, density=1.2)
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('y [mm]')
    ax3.set_title(r'Velocity magnitude $|\mathbf{u}|$ with streamlines')
    ax3.set_aspect('equal')

    # 4. Profiles along centerline: p(x) and jx(x)
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

    # Flux on right axis
    ax4_twin = ax4.twinx()
    color_j = 'tab:blue'
    ax4_twin.set_ylabel(r'$j_x$ [kg/(m²·s)] $\times 10^{3}$', color=color_j)
    ax4_twin.plot(x_line, jx[mid_y, :] * 1000, color=color_j, linewidth=2, linestyle='--', label=r'$j_x$(x)')
    ax4_twin.tick_params(axis='y', labelcolor=color_j)

    ax4.set_title(f'Profiles along y = {Y[mid_y, 0]*1000:.1f} mm')
    ax4.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # Add boundary condition annotations
    fig.text(0.5, 0.01,
             r'BCs: $\rho_W = 1.1$, $\rho_E = 1.0$ (Dirichlet), '
             r'$\partial j/\partial n = 0$ at E/W (Neumann), Periodic in Y',
             ha='center', fontsize=9, style='italic')

    plt.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.08, hspace=0.30, wspace=0.35)
    plt.suptitle('2D FEM Solution: Pressure-Driven Flow (Periodic Channel)', fontsize=12)

    plt.show()


if __name__ == "__main__":
    main()
