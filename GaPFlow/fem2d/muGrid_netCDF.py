"""
Issue combining CartesianDecomposition with FileIONetCDF
"""

from mpi4py import MPI
import numpy as np
import os

from muGrid import (
    CartesianDecomposition,
    FileIONetCDF,
    OpenMode,
    Communicator
)


comm = Communicator(MPI.COMM_WORLD)


def test_decomposition_with_netcdf():
    """Test domain decomposition workflow with NetCDF output."""

    # Domain configuration
    nb_subdivisions = (1, 4)
    nb_ghost_left = (1, 1)
    nb_ghost_right = (1, 1)
    nb_grid_pts = (8, 8)

    # decomposition and field collection
    decomp = CartesianDecomposition(
        comm, nb_grid_pts, nb_subdivisions, nb_ghost_left, nb_ghost_right
    )
    fc = decomp.collection
    field = fc.real_field('solution', 3)

    # Fill interior with test data
    field.p[0, :, :] = np.arange(16).reshape(8, 2) + 1    # rho: 1-16
    field.p[1, :, :] = np.arange(16).reshape(8, 2) + 100  # jx
    field.p[2, :, :] = np.arange(16).reshape(8, 2) + 200  # jy

    # Exchange ghosts (periodic)
    decomp.communicate_ghosts(field)

    print("Field from CartesianDecomposition on rank {}".format(comm.rank))
    print(f"  field.p.shape  (interior): {field.p.shape}")
    print(f"  field.pg.shape (+ ghosts): {field.pg.shape}")
    print(f"  field.shape (internal):    {field.shape}")

    # --- Write to NetCDF using the same fc ---
    # filepath = '/tmp/test_mugrid.nc'  # Works
    # filepath = './test_mugrid.nc'  # Works
    # filepath = './data/test_mugrid.nc'  # Works
    # filepath = './nonexistent_dir/test_mugrid.nc'  # Fails: NC_ENOENT
    filepath = '/home/qd5728/GaPFlow/hans/GaPFlow/fem2d/test_mugrid.nc'  # Test home dir

    file_io = FileIONetCDF(filepath, OpenMode.Overwrite, communicator=comm)
    file_io.register_field_collection(fc, field_names=['solution'])

    # Write frame 0
    print("\nAttempting to write frame...")
    file_io.append_frame().write()

    file_io.close()

    if comm.rank == 0:
        print(f"NetCDF file written: {os.path.getsize(filepath)} bytes")


if __name__ == '__main__':
    test_decomposition_with_netcdf()
