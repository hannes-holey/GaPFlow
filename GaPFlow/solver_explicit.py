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
from .integrate import predictor_corrector, source
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .problem import Problem


class ExplicitSolver:

    def __init__(self, problem: "Problem") -> None:
        self.problem = problem

    def pre_run(self) -> None:
        p = self.problem

        if p.numerics["adaptive"]:
            p.dt = p.numerics["CFL"] * p.dt_crit
        else:
            p.dt = p.numerics['dt']

        p.tol = p.numerics['tol']
        p.max_it = p.numerics['max_it']

    def update(self) -> None:
        """
        Single update iteration performing predictor-corrector for each sweep
        direction and updating constitutive models (pressure, wall/bulk stress).
        """
        p = self.problem

        switch = (p.step % 2 == 0) * 2 - 1 if p.numerics["MC_order"] == 0 else p.numerics["MC_order"]
        directions = [[-1, 1], [1, -1]][(switch + 1) // 2]

        dx = p.grid["dx"]
        dy = p.grid["dy"]
        dt = p.dt

        q0 = p.q.copy()

        one_step_before_output = (p.step + 1) % p.options['write_freq'] == 0

        for i, d in enumerate(directions):
            # update surrogates / constitutive models (predictor on first pass)
            p.pressure.update(predictor=i == 0,
                              compute_var=one_step_before_output)
            p.wall_stress_xz.update(predictor=i == 0,
                                    compute_var=one_step_before_output)
            p.wall_stress_yz.update(predictor=i == 0,
                                    compute_var=one_step_before_output)
            p.bulk_stress.update()

            # fluxes and source terms
            fX, fY = predictor_corrector(
                p.q,
                p.pressure.pressure,
                p.bulk_stress.stress,
                d,
            )

            src = source(
                p.q,
                p.topo.height_and_slopes,
                p.bulk_stress.stress,
                p.wall_stress_xz.lower + p.wall_stress_yz.lower,
                p.wall_stress_xz.upper + p.wall_stress_yz.upper,
            )

            p.q = p.q - dt * (fX / dx + fY / dy - src)

            p._communicate_ghost_buffers()

        # second-order temporal averaging (Crank-Nicolson-like)
        p.q = (p.q + q0) / 2.0

        if p.q_is_valid:
            p.topo.update()
            p.post_update()
        else:
            p.finalize(q0)

    def print_status_header(self) -> None:
        p = self.problem

        if not p.options['silent']:
            print(61 * '-')
            print(f"{'Step':6s} {'Timestep':10s} {'Time':10s} {'CFL':10s} {'Residual':10s}")
            print(61 * '-')
            p.write(params=False)

    def print_status(self, scalars) -> None:
        """
        Write scalars, fields and hyperparameters to disk as configured.
        """
        p = self.problem

        if scalars:
            print(f"{p.step:<6d} {p.dt:.4e} {p.simtime:.4e} {p.cfl:.4e} {p.residual:.4e}")
            p.history["step"].append(p.step)
            p.history["time"].append(p.simtime)
            p.history["ekin"].append(p.kinetic_energy)
            p.history["residual"].append(p.residual)
            p.history["vsound"].append(p.pressure.v_sound)
