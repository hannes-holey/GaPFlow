def _get_MPI_grid(Natoms, size, max_cpu, atoms_per_core=1000):
    """Estimate a suitable MPI processor grid.

    Parameters
    ----------
    Natoms : int
        Total number of atoms
    size : int
        Lateral size parameter
    max_cpu : int
        Maximum available processors
    atoms_per_core : float, optional
        Approximate minimum number of atoms per core (the default is 1000)

    Returns
    -------
    tuple
        Cartesian processor grid (int, int, int)
    """

    ncpus = min(max_cpu, Natoms // atoms_per_core)

    ny = size // 2 + size % 2
    if max_cpu < ny**2:
        ny = 1
        nx = 1
    else:
        nx = ny

    nz = max(ncpus // (nx * ny), 1)

    return (nx, ny, nz)


def write_init(preset="TraPPE", **kwargs):

    if preset == "TraPPE":
        return _write_init_trappe(**kwargs)
    elif preset == "LJ":
        return _write_init_lj(**kwargs)


def _write_init_lj():
    return ""


def _write_init_trappe(cutoff=11., extra_pair="", extra_args="", shift=False, mpi_grid=None):

    out = """
write_once("In Init") {
    # -- Default styles for "TraPPE" --
    units           real
    atom_style      full
    # (Hybrid force field styles were used for portability.)
    bond_style      hybrid harmonic
    angle_style     hybrid harmonic
    dihedral_style  hybrid opls
    improper_style  none
    special_bonds   lj 0.0 0.0 0.0
"""

    # (Original TraPPE has rc=14 A)
    out += f"\tpair_style      hybrid lj/cut {cutoff:.1f}"

    if extra_pair != "lj/cut":
        out += f" {extra_pair} {extra_args}"

    out += "\n\tpair_modify     pair lj/cut mix arithmetic"

    if shift:
        out += " shift yes"

    if mpi_grid is None:
        out += "\nprocessors      1 1 *"
    else:
        out += f"\nprocessors      {mpi_grid[0]} {mpi_grid[1]} {mpi_grid[2]}"

    out += "\n}\n\n"

    return out
