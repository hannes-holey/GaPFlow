#!/usr/bin/env python3
"""
Benchmark script for FEM solver components.

Usage:
    python benchmark.py [--grid 64] [--iterations 10]
    mpirun -n 2 python benchmark.py --grid 128
"""
import resource
resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

import argparse  # noqa: E402
import time  # noqa: E402
import sys  # noqa: E402
import os  # noqa: E402

hans_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, hans_dir)

from mpi4py import MPI  # noqa: E402
from GaPFlow.problem import Problem  # noqa: E402


CONFIG_TEMPLATE = """
options:
    output: /tmp/bench
    write_freq: 1000
    silent: True
grid:
    Lx: 0.1
    Ly: 0.1
    Nx: {nx}
    Ny: {ny}
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
    max_it: 1
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
    linear_solver: {linear_solver}
    R_norm_tol: 1e-08
    pressure_stab_alpha: 100.0
    equations:
        term_list: ['R11x', 'R11y', 'R1T', 'R21x', 'R21y', 'R24x', 'R24y', 'R2Tx', 'R2Ty',
                    'R1Sx', 'R1Sy', 'R1Stabx', 'R1Staby']
"""


def benchmark(nx: int, ny: int, iterations: int, linear_solver: str = 'direct'):
    """Run benchmark for given grid size."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = CONFIG_TEMPLATE.format(nx=nx, ny=ny, linear_solver=linear_solver)
    problem = Problem.from_string(config)
    problem.run()
    solver = problem.solver

    # Warm up
    solver.update_quad()
    M = solver.get_tang_matrix_sparse()
    R = solver.get_residual_vec()
    solver.petsc.assemble(M, R)
    solver.petsc.solve()

    times = {}

    # Benchmark update_quad
    t0 = time.perf_counter()
    for _ in range(iterations):
        solver.update_quad()
    times['update_quad'] = (time.perf_counter() - t0) / iterations

    # Benchmark sparse matrix assembly
    t0 = time.perf_counter()
    for _ in range(iterations):
        M = solver.get_tang_matrix_sparse()
    times['tang_matrix_sparse'] = (time.perf_counter() - t0) / iterations

    # Benchmark residual vector
    t0 = time.perf_counter()
    for _ in range(iterations):
        R = solver.get_residual_vec()
    times['residual_vec'] = (time.perf_counter() - t0) / iterations

    # Benchmark PETSc assembly
    t0 = time.perf_counter()
    for _ in range(iterations):
        solver.petsc.assemble(M, R)
    times['petsc_assemble'] = (time.perf_counter() - t0) / iterations

    # Benchmark PETSc solve
    t0 = time.perf_counter()
    for _ in range(iterations):
        solver.petsc.solve()
    times['petsc_solve'] = (time.perf_counter() - t0) / iterations
    solve_info = solver.petsc.get_convergence_info()

    # Print results on rank 0
    if rank == 0:
        solver_str = f", {linear_solver}" if linear_solver != 'direct' else ''
        print(f"\nBenchmark ({nx}x{ny} grid{problem.decomp.subdomain_info}{solver_str}, {iterations} iterations):")
        print("-" * 55)
        total = sum(times.values())
        for name, t in sorted(times.items(), key=lambda x: -x[1]):
            print(f"  {name:25s}: {t * 1000:8.2f} ms  ({t / total * 100:5.1f}%)")
        print("-" * 55)
        print(f"  {'TOTAL':25s}: {total * 1000:8.2f} ms")
        if linear_solver == 'iterative':
            print(f"\n  KSP iterations: {solve_info['iterations']}, "
                  f"residual: {solve_info['residual_norm']:.2e}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Benchmark FEM solver components')
    parser.add_argument('--grid', type=int, default=128, help='Grid size (NxN)')
    parser.add_argument('--nx', type=int, default=None, help='Grid size in X')
    parser.add_argument('--ny', type=int, default=None, help='Grid size in Y')
    parser.add_argument('--iterations', '-n', type=int, default=10, help='Number of iterations')
    parser.add_argument('--solver', type=str, default='direct',
                        choices=['direct', 'iterative'],
                        help='Linear solver type: direct (MUMPS) or iterative (GMRES+ILU)')
    args = parser.parse_args()

    nx = args.nx if args.nx else args.grid
    ny = args.ny if args.ny else args.grid

    benchmark(nx, ny, args.iterations, args.solver)


if __name__ == "__main__":
    main()
