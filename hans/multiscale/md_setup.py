import os
import numpy as np
import subprocess
import scipy.constants as sci
from ase.lattice.cubic import FaceCenteredCubic

from hans.multiscale.lt_init import write_init, _get_MPI_grid
from hans.multiscale.lt_solid import write_solid_data, write_slab, _create_fcc_wall_ase_rotate
from hans.multiscale.lt_fluid import write_fluid, config_fluid
from hans.multiscale.lt_interface import write_mixing
from hans.multiscale.lt_settings import write_settings
from hans.multiscale.lt_run import write_run, write_restart


def write_template(args, template_dir='moltemplate_files', output_dir="moltemplate_files"):
    """Generate a moltemplate template file (./moltemplate_files/system.lt).

    Moltemplate builds LAMMPS input scripts from the definitions in the template.

    The general structure of the input file is:

    - Init: units, atom_style, interaction_style, MPI domain partitioning
    - Atom Definition: coordinates, moelcule topographies and pair_coeffs
    - Settings: variable/group/... definitions, computes, thermo settings, ...
    - Run: fixes, runs

    System agnostic sections may be included, e.g. from the static subdirectory


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
    solid = args.get("solid", "Au")

    # top wall possibly rotated
    slab_top, nx = _create_fcc_wall_ase_rotate(nx=nx,
                                               ny=ny,
                                               nz=nz,
                                               rotation=target_rotation)

    slab_bot, _ = _create_fcc_wall_ase_rotate(nx=nx,
                                              ny=ny,
                                              nz=nz,
                                              rotation=0.)

    lx, ly, lz = slab_bot.get_cell_lengths_and_angles()[:3]

    num_solid_atoms = slab_bot.get_global_number_of_atoms() + slab_top.get_global_number_of_atoms()

    # fluid
    buffer = 0.1 * lz
    name = args.get("molecule", "pentane")
    molecule_file = os.path.join(template_dir, f"{name}.lt")
    fluid_grid, num_fluid_mol, num_fluid_atoms, initial_gap = config_fluid(molecule_file, lx, ly, target_gap, target_density, buffer=buffer)

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

    restart_file = args.get("restart_file", "run.in.restart")

    moltemplate_command = ["moltemplate.sh",
                           "-overlay-all",
                           "-lammps-script", "run.in.all",
                           "moltemplate_files/system.lt"
                           ]

    subprocess.run(moltemplate_command, shell=False,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # write_restart(restart_file)
