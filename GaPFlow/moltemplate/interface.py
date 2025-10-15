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
