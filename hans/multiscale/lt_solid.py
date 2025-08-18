import numpy as np
from ase.lattice.cubic import FaceCenteredCubic


def write_slab(name='solid', shift=0.):

    # Old version does not work for rotated slabs
    #     out = f"""
    # {name} = new solid [{nx}].move({ax}, 0, 0)
    #                 [{ny}].move(0, {ay}, 0)
    #                 [{nz}].move(0, 0, {az})
    # """

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

    out = "solid {\n\n"

    # Coordinates
    out += "\twrite(\"Data Atoms\") {\n\t\t"
    # data = [f"$atom:au_{i+1} $mol:. @atom:au " + "0.0 " + " ".join(coord)
    #         for i, coord in enumerate(coords)]

    data = []
    offset = 0
    for slab in [slabL, slabU]:

        coords = slab.get_positions()

        data.extend([f"$atom:au_{i+1+offset} $mol:. @atom:au " +
                     f"0.0 {coord[0]:.6e} {coord[1]:.6e} {coord[2]:.6e}"
                     for i, coord in enumerate(coords)])

        offset = len(data)

    out += "\n\t\t".join(data) + "\n\t}\n\n"

    mass = slabL.get_masses()[0]

    # Masses
    out += "\twrite_once(\"Data Masses\") {\n\t\t@atom:au "
    out += f"{mass}"
    out += "\n\t}\n\n"

    # Pair coeffs
    out += "\twrite_once(\"In Settings\") {\n"
    out += pair_coeff_line
    out += "\t\tgroup solid type @atom:au\n\t}\n"
    out += "}\n\n"

    return out


def _create_fcc_wall_ase(size, symbol):

    nx = 4 * size
    ny = 7 * size
    nz = 3

    slab = FaceCenteredCubic(directions=[[1, 1, -2], [-1, 1, 0], [1, 1, 1]],
                             size=(nx, ny, nz),
                             symbol=symbol,
                             pbc=(1, 1, 1))

    ax, ay, az = np.diag(slab.cell)

    # coords = slab.get_positions()
    # slab.positions += np.array([0., 0., az / 6.])

    lx = ax * nx
    ly = ay * ny
    lz = az * nz

    return slab


def _create_fcc_wall_ase_rotate(symbol='Au',
                                a=4.08,
                                ax=[1, 1, 0],
                                ay=[-1, 1, 2],
                                az=[1, -1, 1],
                                rotation=0.,
                                nx=30,
                                ny=None,
                                nz=None,
                                min_angle=4.6,
                                max_angle=6.
                                ):

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
