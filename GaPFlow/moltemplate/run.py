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
