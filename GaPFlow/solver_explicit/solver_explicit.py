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
from ..bc import BoundarySpec, GhostUpdater, sample_bc_spec
from ..logging import get_logger

logger = get_logger("gapflow.run")
if TYPE_CHECKING:
    from ..problem import Problem


class ExplicitSolver:

    def __init__(self, problem: "Problem") -> None:
        self.problem = problem
        self.decomp = problem.decomp
        self.nb_sol = 3

    def build_boundary_conditions(self) -> None:

        specs = []
        no_fun = [None] * 4
        p = self.problem

        self._sol_field = p.fc.get_real_field('solution')

        rho_bc_type, rho_bc_vals = sample_bc_spec(p.grid, 0)
        specs.append(BoundarySpec(p.q[0], 'P1', rho_bc_type,
                                  rho_bc_vals, no_fun, p.decomp,
                                  do_exchange=False))

        jx_bc_type, jx_bc_vals = sample_bc_spec(p.grid, 1)
        specs.append(BoundarySpec(p.q[1], 'P1', jx_bc_type,
                                  jx_bc_vals, no_fun, p.decomp,
                                  do_exchange=False))

        jy_bc_type, jy_bc_vals = sample_bc_spec(p.grid, 2)
        specs.append(BoundarySpec(p.q[2], 'P1', jy_bc_type,
                                  jy_bc_vals, no_fun, p.decomp,
                                  do_exchange=False))

        self.ghost_updater = GhostUpdater(
            decomp=p.decomp,
            problem=p,
            specs=specs,
        )

    def pre_run(self) -> None:
        p = self.problem

        self.build_boundary_conditions()

        if p.numerics["adaptive"]:
            p.dt = p.numerics["CFL"] * p.dt_crit
        else:
            p.dt = p.numerics['dt']

    def update(self) -> None:
        """
        Performs a single time step using the MacCormack [1]_ predictor corrector scheme.

        References
        ----------
        .. [1] MacCormack, R. W. (2003).
               The effect of viscosity in hypervelocity impact cratering
               Journal of Spacecraft and Rockets (reprint)
               https://doi.org/10.2514/2.6901
        """
        p = self.problem

        switch = (p.step % 2 == 0) * 2 - 1 if p.numerics["MC_order"] == 0 else p.numerics["MC_order"]
        directions = [[-1, 1], [1, -1]][(switch + 1) // 2]

        dx = p.grid["dx"]
        dy = p.grid["dy"]
        dt = p.dt

        q0 = p.q.copy()

        # Without active learning, compute variance only before writing
        one_step_before_output = (p.step + 1) % p.options['write_freq'] == 0

        for i, d in enumerate(directions):

            # update surrogates / constitutive models (predictor on first pass)
            p.pressure.update(residuals=p.residual_buffer,
                              predictor=i == 0,
                              compute_var=one_step_before_output,
                              )
            p.wall_stress_xz.update(residuals=p.residual_buffer,
                                    predictor=i == 0,
                                    compute_var=one_step_before_output
                                    )
            p.wall_stress_yz.update(residuals=p.residual_buffer,
                                    predictor=i == 0,
                                    compute_var=one_step_before_output
                                    )
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
                p.topo.full,
                p.bulk_stress.stress,
                p.wall_stress_xz.lower + p.wall_stress_yz.lower,
                p.wall_stress_xz.upper + p.wall_stress_yz.upper,
            )

            p.q[...] = p.q - dt * (fX / dx + fY / dy - src)

            self.decomp.exchange_ghosts(self._sol_field)
            self.ghost_updater.update()

        # second-order temporal averaging (Crank-Nicolson-like)
        p.q[...] = (p.q + q0) / 2.0
        self.decomp.exchange_ghosts(self._sol_field)
        self.ghost_updater.update()

        if p.q_is_valid:
            p.topo.update()
            p._post_update()
        else:
            p._finalize(q0)

    def print_status_header(self) -> None:
        p = self.problem

        if p.options['print_progress']:
            logger.info(61 * '-')
            logger.info(f"{'Step':6s} {'Timestep':10s} {'Time':10s} {'CFL':10s} {'Residual':10s}")
            logger.info(61 * '-')
        self.print_status()
        if p.options['save_output']:
            p.write()

    def print_status(self) -> None:
        """
        Log the current status line, if enabled.
        """
        p = self.problem

        if p.options['print_progress']:
            logger.info(f"{p.step:<6d} {p.dt:.4e} {p.simtime:.4e} {p.cfl:.4e} {p.residual:.4e}")

    def status_record(self) -> dict:
        """
        Scalar values recorded to history.csv for the current step.
        """
        p = self.problem
        return {
            "step": p.step,
            "time": p.simtime,
            "ekin": p.kinetic_energy,
            "residual": p.residual,
            "vsound": p.pressure.v_sound,
        }
