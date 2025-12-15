"""
Test script to verify communicate_ghosts() works correctly with comm.size==1.

This tests that periodic boundary conditions are properly applied via
CartesianDecomposition.communicate_ghosts() even in serial mode.
"""
import numpy as np
from mpi4py import MPI
from muGrid import CartesianDecomposition, Communicator

def test_periodic_wrap_serial():
    """Test that communicate_ghosts applies periodic wrap in serial (size=1)."""

    comm = Communicator(MPI.COMM_WORLD)
    print(f"MPI size: {comm.size}")
    assert comm.size == 1, "This test is designed for serial execution"

    # Create decomposition: 10x1 interior grid with 1 ghost cell on each side
    Nx, Ny = 10, 1
    decomp = CartesianDecomposition(
        comm,
        [Nx, Ny],      # nb_domain_grid_pts (interior)
        [1, 1],        # nb_subdivisions
        [1, 1],        # nb_ghost_left
        [1, 1],        # nb_ghost_right
    )

    fc = decomp.collection
    field = fc.real_field('test', (3,))  # 3 components like solution field

    print(f"field.p.shape (interior): {field.p.shape}")
    print(f"field.pg.shape (with ghosts): {field.pg.shape}")

    # Expected shapes
    assert field.p.shape == (3, Nx, Ny), f"Expected p shape (3, {Nx}, {Ny}), got {field.p.shape}"
    assert field.pg.shape == (3, Nx+2, Ny+2), f"Expected pg shape (3, {Nx+2}, {Ny+2}), got {field.pg.shape}"

    # Initialize interior cells with known values
    # Set component 0 to have distinct values at boundaries
    field.pg[:] = 0.0  # Clear all

    # Interior cells: indices 1 to Nx (in pg coordinates)
    # Set first interior cell (index 1) to 100
    field.pg[0, 1, 1] = 100.0
    # Set last interior cell (index Nx) to 200
    field.pg[0, Nx, 1] = 200.0

    print("\nBefore communicate_ghosts:")
    print(f"  field.pg[0, :, 1] = {field.pg[0, :, 1]}")
    print(f"  Ghost left (index 0): {field.pg[0, 0, 1]}")
    print(f"  First interior (index 1): {field.pg[0, 1, 1]}")
    print(f"  Last interior (index {Nx}): {field.pg[0, Nx, 1]}")
    print(f"  Ghost right (index {Nx+1}): {field.pg[0, Nx+1, 1]}")

    # Apply ghost communication (should apply periodic wrap)
    decomp.communicate_ghosts(field)

    print("\nAfter communicate_ghosts:")
    print(f"  field.pg[0, :, 1] = {field.pg[0, :, 1]}")
    print(f"  Ghost left (index 0): {field.pg[0, 0, 1]}")
    print(f"  First interior (index 1): {field.pg[0, 1, 1]}")
    print(f"  Last interior (index {Nx}): {field.pg[0, Nx, 1]}")
    print(f"  Ghost right (index {Nx+1}): {field.pg[0, Nx+1, 1]}")

    # Check periodic wrap:
    # Left ghost (index 0) should get value from last interior (index Nx)
    # Right ghost (index Nx+1) should get value from first interior (index 1)

    left_ghost = field.pg[0, 0, 1]
    right_ghost = field.pg[0, Nx+1, 1]
    first_interior = field.pg[0, 1, 1]
    last_interior = field.pg[0, Nx, 1]

    print("\nVerification:")
    print(f"  Left ghost should equal last interior: {left_ghost} == {last_interior} ? {np.isclose(left_ghost, last_interior)}")
    print(f"  Right ghost should equal first interior: {right_ghost} == {first_interior} ? {np.isclose(right_ghost, first_interior)}")

    # Assertions
    assert np.isclose(left_ghost, last_interior), \
        f"Periodic wrap failed: left ghost {left_ghost} != last interior {last_interior}"
    assert np.isclose(right_ghost, first_interior), \
        f"Periodic wrap failed: right ghost {right_ghost} != first interior {first_interior}"

    print("\n✓ communicate_ghosts() correctly applies periodic wrap in serial mode!")
    return True


if __name__ == "__main__":
    test_periodic_wrap_serial()
