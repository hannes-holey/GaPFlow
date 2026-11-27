"""Generic quadfield-on-nodes evaluation and imshow helper for GaPFlow.

Usage::

    from GaPFlow.viz.quadfield import evaluate_quadfield_on_nodes, plot_quadfield_on_nodes

    # After problem.run():
    tau = evaluate_quadfield_on_nodes(problem, 'tau_yz_bot')   # (Nx, Ny)
    ax, im = plot_quadfield_on_nodes(problem, 'tau_yz_bot')

Verification snippet (run ad-hoc after problem.run())::

    import numpy as np
    from GaPFlow.viz.quadfield import evaluate_quadfield_on_nodes

    # Cross-check against WallStress reference for a custom yz model
    # (requires patch_registry_for_gp(custom_yz=True) to be in effect):
    tau_yz_bot = evaluate_quadfield_on_nodes(problem, 'tau_yz_bot')
    ref = problem.wall_stress_yz._WallStress__field.pg[3][1:-1, 1:-1]
    np.testing.assert_allclose(tau_yz_bot, ref, rtol=1e-10, atol=0)
    print("OK: tau_yz_bot matches WallStress reference")

    # Newtonian Couette sanity check (flat gap h, no pressure gradient):
    # tau_yz_bot ~ eta * (V_top - V_bot) / h  (verify sign from callable)
    h     = evaluate_quadfield_on_nodes(problem, 'h')
    eta   = evaluate_quadfield_on_nodes(problem, 'eta')
    V_bot = problem.geo['V_bot']
    V_top = problem.geo['V_top']
    expected = eta * (V_top - V_bot) / h
    np.testing.assert_allclose(tau_yz_bot, expected, rtol=1e-6)
    print("OK: tau_yz_bot matches Couette analytic")
"""

import numpy as np

from GaPFlow.solver_implicit.fieldspec import QUAD_FIELD_REGISTRY, resolve_source


def evaluate_quadfield_on_nodes(problem, name: str, *,
                                strip_ghosts: bool = True) -> np.ndarray:
    """Evaluate QUAD_FIELD_REGISTRY entry `name` on the P1 nodal grid.

    Returns shape (Nx, Ny) (interior) when strip_ghosts=True, else the padded
    (Nx+2, Ny+2) array.
    """
    qm = problem.solver.quad_mgr
    ref_shape = qm.nf('rho').shape  # P1 padded (Nx_pad, Ny_pad)

    def _eval(n):
        # 1. QUAD_FIELD_REGISTRY dispatch
        if n in QUAD_FIELD_REGISTRY:
            entry = QUAD_FIELD_REGISTRY[n]
            t = entry['type']
            if t == 'nodal':
                arr = qm.nf(n)
                if n in ('jx', 'jy'):
                    arr = arr[::2, ::2]
                return arr
            if t == 'scalar':
                return np.full(ref_shape, float(resolve_source(problem, entry['source'])))
            if t == 'computed':
                if entry['source'] is None:
                    raise NotImplementedError(
                        f"'{n}' is an inline-computed quadfield with no standalone "
                        f"source (populated by solver-internal methods like "
                        f"_update_squeeze_quad_fields). Read it directly from "
                        f"problem.solver.quad_mgr.quad_fields['{n}'].pg instead."
                    )
                return qm._call_computed(n, _eval)
            raise ValueError(f"unknown registry type {t!r} for {n!r}")

        # 2. P2 momentum variables (live on the fine grid — subsample to P1)
        if n in ('jx', 'jy'):
            if n not in qm.nodal_fields:
                raise ValueError(f"'{n}' is not an active variable in this problem")
            return qm.nf(n)[::2, ::2]

        # 3. P1 nodal fields not in the registry (rho, p, theta if active, E, xi, …)
        if n in qm.nodal_fields:
            return qm.nf(n)

        # 4. Solver-computed quad fields — use first quad-point slice as a 2-D
        #    (Nx_pad, Ny_pad) array.  This covers:
        #      - derivative fields d_dx_X / d_dy_X that the solver computed via
        #        its FEM dx/dy_operator (consistent with what the assembly used)
        #      - solver-managed scalars like theta=0 when cavitation is inactive
        if n in qm.quad_fields:
            return qm.quad_fields[n].pg[0]

        # 5. Derivative fields not yet in quad_fields (unusual — the solver normally
        #    pre-computes them when gap_shear is active).  Fall back to finite
        #    differences on the P1 nodal field as a last resort.
        if n.startswith('d_dx_'):
            base = n[5:]
            if base in qm.nodal_fields:
                return np.gradient(qm.nf(base), qm.dx, axis=0)
        if n.startswith('d_dy_'):
            base = n[5:]
            if base in qm.nodal_fields:
                return np.gradient(qm.nf(base), qm.dy, axis=1)

        raise ValueError(
            f"cannot resolve '{n}': not in QUAD_FIELD_REGISTRY, "
            f"not a P1/P2 nodal field, not a d_dx/d_dy derivative field, "
            f"and not in quad_fields"
        )

    out = _eval(name)
    return out[1:-1, 1:-1] if strip_ghosts else out


def plot_quadfield_on_nodes(problem, name: str, ax=None, *,
                            strip_ghosts: bool = True, **imshow_kwargs):
    """Thin wrapper around evaluate_quadfield_on_nodes + imshow.

    Returns (ax, im). Uses origin='lower', extent=(0.,1.,0.,1.) and transposes
    for (x, y) axes, matching GaPFlow/viz/plotting.py:_plot_sol_from_field_2d.
    """
    import matplotlib.pyplot as plt

    data = evaluate_quadfield_on_nodes(problem, name, strip_ghosts=strip_ghosts)
    if ax is None:
        _, ax = plt.subplots()
    kw = {'origin': 'lower', 'extent': (0., 1., 0., 1.)}
    kw.update(imshow_kwargs)
    im = ax.imshow(data.T, **kw)
    ax.set_xlabel(r'$x/L_x$')
    ax.set_ylabel(r'$y/L_y$')
    ax.set_title(name)
    return ax, im
