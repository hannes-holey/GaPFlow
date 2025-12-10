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
import jax.numpy as jnp
from GaPFlow import Problem


def test_predict_repredict(tmp_path):
    # The choice of unctertainty tolerances should lead to both
    # pressure and shear stress triggering one active learning step.

    sim = f"""
options:
    output: {tmp_path}
    write_freq: 100
    use_tstamp: False
grid:
    Lx: 1470.
    Ly: 1.
    Nx: 200
    Ny: 1
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 0.8
    xW_D: 0.8
geometry:
    type: parabolic
    hmin: 12.
    hmax: 60.
    U: 0.12
    V: 0.
numerics:
    CFL: 0.5
    adaptive: 1
    tol: 1e-8
    dt: 0.05
    max_it: 5_000
properties:
    shear: 2.15
    bulk: 0.
    EOS: BWR
    T: 1.0
    rho0: 0.8
gp:
    press:
        fix_noise: True
        atol: .7
        rtol: 0.
        obs_stddev: 2.e-2
        max_steps: 10
        active_learning: True
    shear:
        fix_noise: True
        atol: .9
        rtol: 0.
        obs_stddev: 4.e-3
        max_steps: 10
        active_learning: True
db:
    # dtool_path: data/train  # defaults to options['output']/train
    init_size: 3
    init_method: rand
    init_width: 0.01 # default (for density)
"""

    testProblem = Problem.from_string(sim)
    testProblem._pre_run()

    for _ in range(3):

        # Remove conditioned GPs for testing
        testProblem.pressure.cond_gp = None
        testProblem.wall_stress_xz.cond_gp = None

        # Builds new conditioned GP (_predict)
        p_mean1, p_var1 = testProblem.pressure._infer_mean_var()
        s_mean1, s_var1 = testProblem.wall_stress_xz._infer_mean_var()

        # Uses cached Cholesky factors (_repredict)
        p_mean2, p_var2 = testProblem.pressure._infer_mean_var()
        s_mean2, s_var2 = testProblem.wall_stress_xz._infer_mean_var()

        assert jnp.isclose(jnp.max(jnp.abs(p_mean1 - p_mean2)), 0.)
        assert jnp.isclose(jnp.max(jnp.abs(p_var1 - p_var2)), 0.)

        assert jnp.isclose(jnp.max(jnp.abs(s_mean1 - s_mean2)), 0.)
        assert jnp.isclose(jnp.max(jnp.abs(s_var1 - s_var2)), 0.)

        testProblem.update()
