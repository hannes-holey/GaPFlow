import os
import numpy as np


def _read_coords_from_lt(file, atom_style='full'):
    """read the coordinates (x,y,z) for a single molecule from an lt file.

    Parameters
    ----------
    file : filename
        str
    atom_style : str, optional
        LAMMPS atom style (the default is 'full', which is currently the only implemented)

    Returns
    -------
    numpy.ndarray
        Atomic coordinates, (natoms, 3)

    Raises
    ------
    RuntimeError
        No valid atom_style given.
    """

    if atom_style not in ["full"]:
        raise RuntimeError('atom_style should be "full"')

    coords = []
    with open(file, 'r') as f:
        for lines in f.readlines():
            line = lines.split()
            if len(line) > 1 and line[0].startswith('$atom:'):
                coord = [float(x) for x in line[4:7]]  # atom_type full
                coords.append(coord)

    coords = np.array(coords)

    return coords


def _get_num_fluid_molecules(name, volume, density):

    mFluid, nC_per_mol = _get_mass_alkane(name)
    Nf = round(density * volume / mFluid)

    return Nf, Nf * nC_per_mol


def config_fluid(file, Lx, Ly, H, density, buffer=25.):
    """Calculate an initial molecule grid given the box dimensions and
    adjust the gap height for the initial setup to fit all molecules
    without overlap.


    Parameters
    ----------
    file : str
        Molecule topography filename
    Lx : float
        Box dimension x
    Ly : float
        Box dimension y
    H : float
        Target gap height
    density: float
        Target fluid density
    buffer: float
        "Safety distance" between the outermost fluid layer and the wall 

    Returns
    -------
    tuple
        Molecule grid
    int
        Number of fluid molecules
    int
        Number of fluid atoms
    float
        Initial gap height 

    Raises
    ------
    RuntimeError
        Lateral box size too small for molecule
    """

    name = file.split(os.sep)[-1].split('.')[0]

    volume = Lx * Ly * H
    num_fluid_mol, num_fluid_atoms = _get_num_fluid_molecules(name, volume, density)

    coords = _read_coords_from_lt(file)
    lx, ly, lz = coords.max(0) - coords.min(0)

    # number of molecules in x and y
    nxf = int(np.floor(Lx / (2 * lx)))
    nyf = int(np.floor(Ly / (2 * ly)))

    if nxf == 0 or nyf == 0:
        raise RuntimeError("Molecule larger than specified box. Increase box size!")

    max_molecules_per_plane = nxf * nyf

    # number of molecules in z
    nzf = num_fluid_mol // max_molecules_per_plane
    if num_fluid_mol % max_molecules_per_plane != 0:
        nzf += 1

    lz = max(lz, 1.5)
    initial_gap = max(2. * nzf * lz, H) + 2 * buffer

    return (nxf, nyf, nzf), num_fluid_mol, num_fluid_atoms, initial_gap


def _get_mass_alkane(name):
    """Get molar mass of an alkane molecule.

    Parameters
    ----------
    name : str
        Name of the molecule

    Returns
    -------
    float
        Molar mass in g/mole
    int
        Number of pseudo (C) atoms
    """

    molecules = {'pentane': [3, 2, 0],
                 'decane': [8, 2, 0],
                 'hexadecane': [14, 2, 0], }

    assert name in molecules.keys()

    nCH2, nCH3, nCH4 = molecules[name]

    # United Atom pseudo particles
    mCH2 = 14.1707
    mCH3 = 15.2507
    mCH4 = 16.3307

    return nCH2 * mCH2 + nCH3 * mCH3 + nCH4 * mCH4, np.sum(molecules[name])


def write_fluid(name, Nf, mol_grid, slab_size, gap, buffer=25.):

    Nx, Ny, Nz = mol_grid
    Lx, Ly, Lz = slab_size

    out = f"""
import {name}.lt
"""

    # Nfluid = round(sci.N_A * density * lx * ly * gap * 1.e-24 / M)

    name = name.split('.')[0]
    ax = Lx / Nx
    ay = Ly / Ny
    az = (gap - 2 * buffer) / Nz

    out += f"""
fluid = new {name} [{Nx}].move({ax}, 0.0, 0.0)
                 [{Ny}].move(0.0, {ay}, 0.0)
                 [{Nz}].move(0.0, 0.0, {az})

fluid[*][*][*].move(0, 0, {Lz + buffer})
"""
    # i = 0
    # diff = 0
    # while diff < delta:
    #     out += f"delete fluid[0-{min(Nx, delta-diff-1)}][{i}][0]\n"
    #     i += 1
    #     diff += Nx

    delta = Nx * Ny * Nz - Nf
    for i in range(Nx):
        for j in range(Ny):
            if delta == 0:
                break
            else:
                out += f"delete fluid[{i}][{j}][0]\n"
                delta -= 1

    box_offset = 10.

    out += "\nwrite_once(\"Data Boundary\") {\n\t"
    out += f"0. {Lx:.4f} xlo xhi\n\t"
    out += f"0. {Ly:.4f} ylo yhi\n\t"
    out += f"0. {2 * Lz + gap + box_offset:.4f} zlo zhi\n"
    out += "}\n"

    return out
