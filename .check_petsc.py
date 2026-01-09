#!/usr/bin/env python3
"""
Verify PETSc and petsc4py installation for GaPFlow.

Usage:
    python3 .check_petsc.py
"""
import sys


def check_petsc():
    """Check PETSc installation and print diagnostic info."""
    print("PETSc Installation Check")
    print("=" * 40)

    # Check petsc4py import
    try:
        import petsc4py
        from petsc4py import PETSc
        print("petsc4py:      OK")
    except ImportError as e:
        print(f"petsc4py:      FAILED ({e})")
        print("\nPETSc is not installed or not properly configured.")
        print("Run: bash install_petsc.sh")
        return False

    # Version info
    version = PETSc.Sys.getVersion()
    print(f"PETSc version: {version[0]}.{version[1]}.{version[2]}")
    print(f"petsc4py ver:  {petsc4py.__version__}")

    # Check MPI
    comm = PETSc.COMM_WORLD
    print(f"MPI size:      {comm.getSize()}")
    print(f"MPI rank:      {comm.getRank()}")

    # Check available solvers
    print()
    print("Linear Solvers")
    print("-" * 40)

    # Test KSP creation
    try:
        ksp = PETSc.KSP().create()
        print("KSP create:    OK")
        ksp.destroy()
    except Exception as e:
        print(f"KSP create:    FAILED ({e})")
        return False

    # Check for MUMPS (direct solver)
    try:
        mat = PETSc.Mat().create()
        mat.setSizes([10, 10])
        mat.setType('aij')
        mat.setUp()

        ksp = PETSc.KSP().create()
        ksp.setOperators(mat)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setFromOptions()
        print("MUMPS solver:  OK")
        ksp.destroy()
        mat.destroy()
    except Exception as e:
        print(f"MUMPS solver:  NOT AVAILABLE ({e})")
        print("  (MUMPS is recommended for direct solves)")

    # Check iterative solver
    try:
        mat = PETSc.Mat().create()
        mat.setSizes([10, 10])
        mat.setType('aij')
        mat.setUp()

        ksp = PETSc.KSP().create()
        ksp.setOperators(mat)
        ksp.setType('bcgs')
        pc = ksp.getPC()
        pc.setType('ilu')
        ksp.setFromOptions()
        print("BiCGSTAB+ILU:  OK")
        ksp.destroy()
        mat.destroy()
    except Exception as e:
        print(f"BiCGSTAB+ILU:  FAILED ({e})")

    print()
    print("=" * 40)
    print("PETSc is correctly installed!")
    print()

    # Check GaPFlow integration
    try:
        from GaPFlow import HAS_PETSC
        if HAS_PETSC:
            print("GaPFlow HAS_PETSC: True")
        else:
            print("GaPFlow HAS_PETSC: False (unexpected)")
    except ImportError:
        print("GaPFlow not installed in this environment")

    return True


if __name__ == "__main__":
    success = check_petsc()
    sys.exit(0 if success else 1)
