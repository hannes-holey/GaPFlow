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
