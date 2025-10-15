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
import scipy.constants as sci


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
