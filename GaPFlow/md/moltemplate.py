#
# Copyright 2025 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import numpy as np
import subprocess
import scipy.constants as sci
from ase.lattice.cubic import FaceCenteredCubic

from .utils import _get_MPI_grid


def write_init(preset="TraPPE", **kwargs):

    if preset == "TraPPE":
        return _write_init_trappe(**kwargs)
    elif preset == "LJ":
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


def write_slab(name='solid', shift=0.):
    """Write the moltemplate input for the two wall slabs

    Parameters
    ----------
    name : str, optional
        The name (the default is 'solid')
    shift : float, optional
         A vertical (z) shift applied to all coordinates (the default is 0.)
    """

    out = f"""
{name} = new {name}[0][0][0]
"""

    out += f"\n{name}[*][*][*].move(0., 0., {shift})\n"

    return out


def write_solid_data(slabL,
                     slabU,
                     pair_style="eam",
                     eps=5.29,
                     sig=2.629
                     ):

    # Coordinates
    out = "solid {\n\n"
    out += "\twrite(\"Data Atoms\") {\n\t\t"
    data = []
    offset = 0
    for slab in [slabL, slabU]:

        coords = slab.get_positions()

        data.extend([f"$atom:au_{i + 1 + offset} $mol:. @atom:au "
                     + f"0.0 {coord[0]:.6e} {coord[1]:.6e} {coord[2]:.6e}"  # noqa: W503
                     for i, coord in enumerate(coords)])

        offset = len(data)

    out += "\n\t\t".join(data) + "\n\t}\n\n"

    # Masses
    mass = slabL.get_masses()[0]
    out += "\twrite_once(\"Data Masses\") {\n\t\t@atom:au "
    out += f"{mass}"
    out += "\n\t}\n\n"

    # Pair coeffs
    if pair_style == "eam":
        file = "static/Au_u3.eam"
        pair_coeff_line = f"\t\tpair_coeff @atom:au @atom:au eam {file}\n"
    elif pair_style == "eam/alloy":
        file = "static/Au-Grochola-JCP05.eam.alloy"
        pair_coeff_line = f"\t\tpair_coeff * * eam/alloy {file} Au NULL NULL NULL \n"
    elif pair_style == "lj/cut":
        # dafults from Heinz et al., J. Phys. Chem. C 112 2008
        pair_coeff_line = f"\t\tpair_coeff @atom:au @atom:au {eps} {sig}\n"
    else:
        pair_coeff_line = ""

    out += "\twrite_once(\"In Settings\") {\n"
    out += pair_coeff_line
    out += "\t\tgroup solid type @atom:au\n\t}\n"
    out += "}\n\n"

    return out


def _create_fcc_wall_ase(symbol='Au',
                         a=4.08,
                         ax=[1, 1, 0],
                         ay=[-1, 1, 2],
                         az=[1, -1, 1],
                         rotation=0.,
                         nx=30,
                         ny=None,
                         nz=7,
                         min_angle=4.4,
                         max_angle=6.
                         ):
    """Create a slab of face-centered cubic atoms using ASE.

    Parameters
    ----------
    symbol : str, optional
        Element symbol (the default is 'Au')
    a : float, optional
        Lattice parameter (the default is 4.08, for gold)
    ax : list, optional
        Lattice vector pointing in x direction (the default is [1, 1, 0])
    ay : list, optional
        Lattice vector pointing in y direction (the default is [-1, 1, 2])
    az : list, optional
        Lattice vector pointing in z direction (the default is [1, -1, 1])
    rotation : float, optional
        Rotation angle around the y axis in degreees (the default is 0.)
    nx : int, optional
        Number of repetitions in x direction (the default is 30)
    ny : int or None, optional
        Number of repetitions in y direction (the default is None, which
        will create a nearly quadratic (x,y)-shape)
    nz : int, optional
        Number of repetitions in z direction (the default is 7)
    min_angle : float, optional
        Minimum angle in degrees to apply a rotation (the default is 4.4)
    max_angle : float, optional
        Maximum angle in degrees to apply a rotation (the default is 6.)

    Returns
    -------
    ase.Atoms
        The fcc slab
    int
        Number of repetitions used in x direction.
    """

    if abs(rotation) < min_angle:
        rotation = None
    elif abs(rotation) > max_angle:
        raise RuntimeError("Only small rotations possible")

    # for 111 surfaces (110 sliding)
    lx0 = np.sqrt(2) / 2. * a
    ly0 = np.sqrt(6) / 2 * a
    lz0 = np.sqrt(3) * a

    if rotation is not None:
        nx = abs(int(np.floor(lz0 / lx0 / np.tan(rotation / 180. * np.pi))))

    if ny is None:
        ny = int((lx0 * nx) / ly0)
    if nz is None:
        nz = 7

    # Create FCC lattice
    fcc = FaceCenteredCubic(
        # directions=[[1, 1, -2], [-1, 1, 0], [1, 1, 1]],  # my box
        # size=(20, 40, nz),
        directions=[ax, ay, az],  # Andrea's box
        size=(nx, ny, nz),
        symbol='Au',
        pbc=(1, 1, 1))

    if rotation is not None:
        cell = fcc.get_cell()
        lx, ly, lz = np.diag(cell)
        print('Cell size: ', lx, ly, lz)

        # Rotate particle coordinates
        alpha_rad = np.sign(rotation) * np.arctan(lz0 / nx / lx0)
        alpha = alpha_rad * 180 / np.pi
        print("Rotation angle (y):", alpha)
        fcc.rotate(alpha, 'y',
                   # center=fcc.get_center_of_mass(),  # (lx / 2., ly / 2., lz / 2.)
                   )

        # Apply simple shear in xz plane equivalent to (small) rotation
        gamma = np.tan(alpha_rad)
        M = np.array([[1., 0, -gamma],
                      [0., 1, 0],
                      [0, 0, 1]])

        coords = fcc.get_positions()
        coords_transform = np.einsum('ij,...j->...i', M, coords)
        fcc.positions = coords_transform

        # Wrap coordinate into orthorhombic box
        fcc.wrap()

    return fcc, nx


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


def write_mixing():

    # TODO: read pair_coeffs for mixing, e.g., from trappe1998.lt

    out = "\nwrite_once(\"In Settings\"){"

    out += r"""

    variable    eps_Au equal 5.29
    variable    sig_Au equal 2.629

    variable    eps_CH2 equal 0.091411522
    variable    eps_CH3 equal 0.194746286
    variable    eps_CH4 equal 0.294106636
    variable    sig_CH2 equal 3.95
    variable    sig_CH3 equal 3.75
    variable    sig_CH4 equal 3.73

    variable    eps_CH2_Au equal sqrt(v_eps_CH2*v_eps_Au)
    variable    eps_CH3_Au equal sqrt(v_eps_CH3*v_eps_Au)
    variable    eps_CH4_Au equal sqrt(v_eps_CH4*v_eps_Au)
    variable    sig_CH2_Au equal (v_sig_CH2+v_sig_Au)/2.
    variable    sig_CH3_Au equal (v_sig_CH3+v_sig_Au)/2.
    variable    sig_CH4_Au equal (v_sig_CH4+v_sig_Au)/2.

    # Mixed interactions
    pair_coeff @atom:solid/au @atom:TraPPE/CH2 lj/cut \$\{eps_CH2_Au\} \$\{sig_CH2_Au\}
    pair_coeff @atom:solid/au @atom:TraPPE/CH3 lj/cut \$\{eps_CH3_Au\} \$\{sig_CH3_Au\}
    pair_coeff @atom:solid/au @atom:TraPPE/CH4 lj/cut \$\{eps_CH4_Au\} \$\{sig_CH4_Au\}

"""

    out += "}\n"

    return out


def write_settings(args):

    # FIXME: not hardcoded
    # effective wall fluid distance / hardcoded for TraPPE / gold
    # (You slightly miss the target gap height without it)
    offset = (3.75 + 2.63) / 2.

    density_real = args.get("density")  # g / mol / A^3
    density_SI = density_real / (sci.N_A * 1e-24)

    U_SI = args.get("vWall")
    U_real = U_SI * 1e-5  # m/s to A/fs

    h = args.get("gap_height")

    nlayers = 9  # 3 * unit cell size (default)
    nthermal = (nlayers - 1) // 2 + (nlayers - 1) % 2

    # Couette flow
    couette = args.get("couette", False)
    #
    if couette:
        jx_SI = density_SI * U_SI / 2. * 1e3  # kg / m^2 s
        jx_real = jx_SI * sci.N_A * 1e-32  # g/mol/A^2/fs
        jy_real = 0.
    else:
        jx_real = args.get("fluxX")
        jy_real = args.get("fluxY")

    timestep = args.get("timestep", 1.)
    Ninit = args.get("Ninit", 50_000)
    Nsteady = args.get("Nsteady", 100_000)  # should depend on sliding velocity and size
    Nsample = args.get("Nsample", 300_000)
    temperature = args.get("temperature", 300.)

    nbinz = args.get("nbinz", 200)
    Nevery = args.get("Nevery", 10)
    Nrepeat = args.get("Nrepeat", 100)
    Nfreq = args.get("Nfreq", 1000)
    dumpfreq = args.get("Nfreq", 10_000)

    rotation = args.get("rotation", 0.)
    if abs(rotation) > 4.:
        angle_sf = 1.99
    else:
        angle_sf = 1.

    out = "\nwrite_once(\"In Settings\"){"
    out += f"""

    variable        offset equal {offset}  # mismatch between initial and target gap

    variable        dt equal {timestep}
    variable        Ninit equal {Ninit}
    variable        Nsteady equal {Nsteady}
    variable        Nsample equal {Nsample}

    variable        input_fluxX equal {jx_real}
    variable        input_fluxY equal {jy_real}
    variable        input_temp equal {temperature} # K
    variable        vWall equal {U_real} # A/fs
    variable        hmin equal {h}

    # Wall sections
    variable        nwall equal 3
    variable        ntherm equal {nthermal}
    variable        angle_sf equal {angle_sf}

    # sampling // spatial
    variable        nbinz index {nbinz}

    # sampling // temporal
    variable        Nevery equal {Nevery}
    variable        Nrepeat equal {Nrepeat}
    variable        Nfreq equal {Nfreq}

    variable        dumpfreq equal {dumpfreq}


    include         static/in.settings.lmp

"""
    out += "}\n"

    return out


def write_run():

    # TODO: option to restart simulation

    out = """
write_once("In Run"){

    include static/in.run.min.lmp
    include static/in.run.equil.lmp
    include static/in.run.steady.lmp
    include static/in.run.sample.lmp

}
"""

    return out


def write_restart(restart_file):
    s = f"""
# ----------------- Load restart file -----------------

read_restart "{restart_file}"

# ----------------- Settings Section -----------------

include "system.in.settings"

# ----------------- Run Section -----------------

include "static/in.flow.lmp"
include "static/in.run.sample.lmp"
"""

    with open("run.in.restart", "w") as f:
        f.write(s)


def write_template(args, template_dir='moltemplate_files', output_dir="moltemplate_files"):
    """Generate a moltemplate template file (./moltemplate_files/system.lt).

    Moltemplate builds LAMMPS input scripts from the definitions in the template.

    The general structure of the input file is:

    - Init: units, atom_style, interaction_style, MPI domain partitioning
    - Atom Definition: coordinates, moelcule topographies and pair_coeffs
    - Settings: variable/group/... definitions, computes, thermo settings, ...
    - Run: fixes, runs

    System agnostic sections may be included, e.g. from the static subdirectory.


    Parameters
    ----------
    args : dict
        Dictionary

    Returns
    -------
    int
        Number of MPI processes for MD run
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # general
    shift = args.get("shift", False)
    max_cpu = args.get("ncpu")
    wall_potential = args.get("wall", "eam/alloy")

    # input variables
    target_density = args.get("density")  # g/mol/ A^3
    # target_density *= sci.N_A * 1e-24    # g / cm^3 to g/mol/A^3
    target_gap = args.get("gap_height")  # Angstrom
    target_rotation = args.get("rotation", 0.)

    mpi_grid = args.get("mpiGrid", None)

    # solid, create ASE Atoms object
    nx = args.get("nx", 21)
    ny = args.get("ny", None)
    nz = args.get("nz", None)
    # solid = args.get("solid", "Au")

    # top wall possibly rotated
    slab_top, nx = _create_fcc_wall_ase(nx=nx,
                                        ny=ny,
                                        nz=nz,
                                        rotation=target_rotation)

    slab_bot, _ = _create_fcc_wall_ase(nx=nx,
                                       ny=ny,
                                       nz=nz,
                                       rotation=0.)

    lx, ly, lz = slab_bot.get_cell_lengths_and_angles()[:3]

    num_solid_atoms = slab_bot.get_global_number_of_atoms() + slab_top.get_global_number_of_atoms()

    # fluid
    buffer = 0.1 * lz
    name = args.get("molecule", "pentane")
    molecule_file = os.path.join(template_dir, f"{name}.lt")
    fluid_grid, num_fluid_mol, num_fluid_atoms, initial_gap = config_fluid(
        molecule_file, lx, ly, target_gap, target_density, buffer=buffer)

    # move top wall up
    slab_top.positions += np.array([0., 0., lz + initial_gap])

    # Settings
    Natoms = num_fluid_atoms + num_solid_atoms

    if mpi_grid is None:
        mpi_grid = _get_MPI_grid(Natoms, nx // 7, max_cpu)

    outfile = os.path.join(output_dir, 'system.lt')
    with open(outfile, 'w') as f:

        # Init
        f.write(write_init(extra_pair=wall_potential, shift=shift, mpi_grid=mpi_grid))

        # Atom definition // Data
        # Write solid
        f.write(write_solid_data(slab_bot, slab_top, pair_style=wall_potential))
        f.write(write_slab(name='solid'))

        # Write fluid
        f.write(write_fluid(name, num_fluid_mol, fluid_grid, (lx, ly, lz), initial_gap, buffer=buffer))

        # Write interface
        if wall_potential != "lj/cut":  # lj/cut
            f.write(write_mixing())

        # Write settings
        f.write(write_settings(args))

        # Write run
        f.write(write_run())

    return np.prod(mpi_grid)


def build_template(args):

    # restart_file = args.get("restart_file", "run.in.restart")

    moltemplate_command = ["moltemplate.sh",
                           "-overlay-all",
                           "-lammps-script", "run.in.all",
                           "moltemplate_files/system.lt"
                           ]

    subprocess.run(moltemplate_command, shell=False,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # write_restart(restart_file)
