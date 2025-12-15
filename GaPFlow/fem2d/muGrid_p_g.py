import muGrid
from mpi4py import MPI

Nx, Ny = 4, 4

comm = muGrid.Communicator(MPI.COMM_WORLD)
nb_processes = comm.size
nb_subdivisions = (1,4)
nb_ghost_left = (1, 1)
nb_ghost_right = (1, 1)
nb_grid_pts=(Nx, Ny)

decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, nb_subdivisions=nb_subdivisions, 
                                              nb_ghost_left=nb_ghost_left, nb_ghost_right=nb_ghost_right)
fc = decomposition.collection
xg, yg = decomposition.coordsg

field = fc.real_field("test-field")
field.p[:] += comm.rank
decomposition.communicate_ghosts(field)

# overwrite ghost data for demonstration purposes
field.pg[:, -1] = -1

print(f"[Rank {comm.rank}] muGrid owns local shape: {field.p.shape}")
print(f"[Rank {comm.rank}] muGrid field data:\n{field.pg}")

decomposition.communicate_ghosts(field)
print(f"[Rank {comm.rank}] muGrid field data after communication:\n{field.pg}")
