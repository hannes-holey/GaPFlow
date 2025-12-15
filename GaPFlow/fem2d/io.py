from mpi4py import MPI
import numpy as np
import muGrid

comm = muGrid.Communicator(MPI.COMM_WORLD)

nb_subdivisions = (2, 2)
nb_domain_grid_pts = (5, 5)  # test uses 5, not 8
nb_ghost_left = (1, 1)
nb_ghost_right = (2, 2)

cart_decomp = muGrid.CartesianDecomposition(
    comm,
    list(nb_domain_grid_pts),
    list(nb_subdivisions),
    list(nb_ghost_left),
    list(nb_ghost_right),
)

field = cart_decomp.collection.real_field("test_field")
field.pg = (cart_decomp.icoordsg**2).sum(axis=0)

filename = "/tmp/test_io_output.nc"
f = muGrid.FileIONetCDF(filename, muGrid.OpenMode.Overwrite, comm)
f.register_field_collection(cart_decomp.collection)  # No field_names!
f.append_frame().write()
f.close()

if comm.rank == 0:
    print("Success!")