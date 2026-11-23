
import numpy as np

from ..terms import Term

terms = []
for wall in ['top', 'bot']:
    for dir in ['x', 'y']:
        # -1 because ustop = Utop-Uftop ,
        # 1 because usbot = Ufbot - U
        #
        # This way slip velocities are both  positive for couette flow
        sign_us = -1  if wall == 'top' else 1
        tauname = f'tau_slip_{wall}_{dir}'
        terms.append(Term(
            name=f'Rslip_slip{dir}',
            description=f'slip stress {dir} on {wall} wall',
            res=f'wallstress_{wall}_{dir}',
            dep_vars=[f'Uf_{wall}', f'Vf_{wall}'],
            dep_vals=[tauname, f'd{tauname}_dUs', f'd{tauname}_dVs'],
            fun=lambda ctx, _t=tauname: lambda u, v: ctx[_t](),
            # the extra arguments with default values are a trick so that the values at definition time are eveluated, and not the values at call time
            der_funs=[lambda ctx, _s=sign_us, _t=tauname: lambda u, v: _s * ctx[f'd{_t}_dUs'](),  # for top wall us = Vtop - Vf so that d / DVf = - d/ dvs
                      lambda ctx, _s=sign_us, _t=tauname: lambda u, v: _s * ctx[f'd{_t}_dVs']()]
        ))

        # Fluid stress with a negative sign so both end up equals
        tauname_f = f'tau_{dir}z_{wall}'
        vars = [f'Uf_{wall}', f'Vf_{wall}', f'j{dir}']
        terms.append(Term(
            name=f'Rslip_fluid{dir}',
            description=f'fluid stress {dir} on {wall} wall',
            res=f'wallstress_{wall}_{dir}',
            dep_vars=vars,  # I assume The list does not need to be exhaustive, we can focus on the main effects
            dep_vals=[tauname_f] + [f'd{tauname_f}_d{v}' for v in vars],
            fun=lambda ctx, _t=tauname_f: lambda *args: -ctx[_t](),
            der_funs=[lambda ctx, _t=tauname_f, _v=v: lambda *args: -ctx[f'd{_t}_d{_v}']() for v in vars]
        ))


# Additions to the wall stress equations: derivatives with respect to the fluid top and bottom velocities
#
#
# R24x = Term(
#     name='R24x',
#     description='wall stress x',
#     res='momentum_x',
#     dep_vars=['p', 'jx'],
#     dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx', 'dtau_xz_dh', 'drho_dp'],
#     fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
#     der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho']() * ctx['drho_dp'](),
#               lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx']()],
#     der_h=lambda ctx: lambda *args: -1 / ctx['h']() ** 2 * ctx['tau_xz']() + 1 / ctx['h']() * ctx['dtau_xz_dh']())
#
# R24y = Term(
#     name='R24y',
#     description='wall stress y',
#     res='momentum_y',
#     dep_vars=['p', 'jy'],
#     dep_vals=['h', 'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy', 'dtau_yz_dh', 'drho_dp'],
#     fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_yz'](),
#     der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_drho']() * ctx['drho_dp'](),
#               lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_djy']()],
#     der_h=lambda ctx: lambda *args: -1 / ctx['h']() ** 2 * ctx['tau_yz']() + 1 / ctx['h']() * ctx['dtau_yz_dh']())