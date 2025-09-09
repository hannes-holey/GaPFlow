import pytest
import numpy as np

from GaPFlow.models.velocity_profile import get_velocity_profiles


@pytest.mark.parametrize('slip', ['both', 'top', 'bottom', 'none'])
def test_flow_rate(slip):

    Nz = 10_000
    hmax = 2.

    z = np.linspace(0., hmax, Nz)
    q = np.array([1., 2., 1.])

    Ls = 0.5

    u, v = get_velocity_profiles(z, q, Ls=Ls, U=1., V=1., slip=slip)

    assert np.isclose(np.trapezoid(u, z) / hmax, q[1])
    assert np.isclose(np.trapezoid(v, z) / hmax, q[2])
