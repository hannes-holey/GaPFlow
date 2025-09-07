import os
from datetime import datetime
import yaml


def print_header(s, n=60, f0='*', f1=' '):

    if len(s) > n:
        n = len(s) + 4

    w = n + len(s) % 2
    b = (w - len(s)) // 2 - 1
    print(w * f0)
    print(f0 + b * f1 + s + b * f1 + f0)
    print(w * f0)


def print_dict(d):
    for k, v in d.items():
        print(f'  - {k:<25s}: {v}')


def create_output_directory(name):

    timestamp = datetime.now().replace(microsecond=0).strftime("%Y-%m-%d_%H%M%S")
    outbase = os.path.dirname(name)
    outname = timestamp + '_' + os.path.basename(name)
    outdir = os.path.join(outbase, outname)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        if len(os.listdir(outdir)) > 0:
            raise RuntimeError('Output path exists and is not empty.')

    print_header(f"Writing output into: {outdir}", f0=' ', f1=' ')

    return outdir


def write_yaml(output_dict, fname):

    with open(fname, 'w') as FILE:
        yaml.dump(output_dict, FILE)


def read_yaml_input(file):

    print_header("PROBLEM SETUP")

    print(f'Reading input file: {file}')

    sanitizing_functions = {'options': sanitize_options,
                            'grid': sanitize_grid,
                            'geometry': sanitize_geometry,
                            'numerics': sanitize_numerics,
                            'properties': sanitize_properties,
                            'gp': sanitize_gp}

    sanitized_dict = {}

    with open(file, 'r') as ymlfile:
        raw_dict = yaml.full_load(ymlfile)

        for key, value in raw_dict.items():

            if key in sanitizing_functions.keys():
                print(f'- {key}:')
                sanitized_dict[key] = sanitizing_functions[key](raw_dict[key])

    print_header("PROBLEM SETUP COMPLETED")

    return sanitized_dict


def sanitize_options(d):
    out = {}
    out['output'] = str(d.get('output', 'example'))
    out['write_freq'] = int(d.get('write_freq', 1000))

    print_dict(out)

    return out


def sanitize_grid(d):

    out = {}

    # x
    out['Nx'] = int(d.get('Nx', 100))
    if 'Lx' in d.keys():
        out['Lx'] = float(d.get('Lx', 1.))
        out['dx'] = out['Lx'] / out['Nx']
    elif 'dx' in d.keys():
        out['dx'] = float(d.get('dx', 0.1))
        out['Lx'] = out['dx'] * out['Nx']
    else:
        raise IOError("Must specify grid size (Nx) with either dx or Lx.")

    # y
    out['Ny'] = int(d.get('Ny', 1))
    if 'Ly' in d.keys():
        out['Ly'] = float(d.get('Ly', 1.))
        out['dy'] = out['Ly'] / out['Ny']
    elif 'dy' in d.keys():
        out['dy'] = float(d.get('dy', 0.1))
        out['Ly'] = out['dy'] * out['Ny']
    else:
        raise IOError("Must specify grid size (Ny) with either dy or Ly.")

    # x BCs
    bc_xE = list(d.get('xE', ['P', 'P', 'P']))
    bc_xW = list(d.get('xW', ['P', 'P', 'P']))

    assert all([b in ['P', 'N', 'D'] for b in bc_xE])
    assert all([b in ['P', 'N', 'D'] for b in bc_xW])

    out['bc_xE_P'] = [b == 'P' for b in bc_xE]
    out['bc_xE_D'] = [b == 'D' for b in bc_xE]
    out['bc_xE_N'] = [b == 'N' for b in bc_xE]
    out['bc_xW_P'] = [b == 'P' for b in bc_xW]
    out['bc_xW_D'] = [b == 'D' for b in bc_xW]
    out['bc_xW_N'] = [b == 'N' for b in bc_xW]

    if any(out['bc_xE_D']):
        out['bc_xE_D_val'] = d.get('xE_D', 1.)
        if out['bc_xE_D_val'] is None:
            raise IOError("Need to specify Dirichlet BC value")

    if any(out['bc_xW_D']):
        out['bc_xW_D_val'] = d.get('xW_D', 1.)
        if out['bc_xW_D_val'] is None:
            raise IOError("Need to specify Dirichlet BC value")

    assert all([e == w for e, w in zip(out['bc_xE_P'], out['bc_xW_P'])])

    # y BCs
    bc_yS = list(d.get('yS', ['P', 'P', 'P']))
    bc_yN = list(d.get('yN', ['P', 'P', 'P']))

    assert all([b in ['P', 'N', 'D'] for b in bc_yS])
    assert all([b in ['P', 'N', 'D'] for b in bc_yN])

    out['bc_yS_P'] = [b == 'P' for b in bc_yS]
    out['bc_yS_D'] = [b == 'D' for b in bc_yS]
    out['bc_yS_N'] = [b == 'N' for b in bc_yS]
    out['bc_yN_P'] = [b == 'P' for b in bc_yN]
    out['bc_yN_D'] = [b == 'D' for b in bc_yN]
    out['bc_yN_N'] = [b == 'N' for b in bc_yN]

    if any(out['bc_yS_D']):
        out['bc_yS_D_val'] = d.get('yS_D', None)
        if out['bc_yS_D_val'] is None:
            raise IOError("Need to specify Dirichlet BC value")

    if any(out['bc_yN_D']):
        out['bc_yN_D_val'] = d.get('xW_D', None)
        if out['bc_yN_D_val'] is None:
            raise IOError("Need to specify Dirichlet BC value")

    assert all([s == n for s, n in zip(out['bc_yS_P'], out['bc_yN_P'])])

    print_dict(out)

    return out


def sanitize_geometry(d):

    available = ['journal']
    out = {}

    out['U'] = float(d.get('U', 1.))
    out['V'] = float(d.get('V', 0.))
    out['type'] = str(d.get('type', 'none'))

    if out['type'] not in available:
        raise IOError("Specify a valid geometry type")

    if out['type'] == 'journal':
        if "CR" and 'eps' in d.keys():
            out["CR"] = float(d.get("CR"))
            out["eps"] = float(d.get("eps"))
        elif "hmin" and 'hmax' in d.keys():
            out["hmin"] = float(d.get("hmin"))
            out["hmax"] = float(d.get("hmax"))
        else:
            raise IOError("Need to specify either clearance ratio and eccentrity or min/max gap height")

    print_dict(out)

    return out


def sanitize_properties(d):

    out = {}

    # Viscsosities
    out['shear'] = float(d.get('shear', -1.))
    if out['shear'] < 0.:
        raise IOError("Specify a a (non-negative) shear viscosity")
    out['bulk'] = float(d.get('bulk', -1.))

    # EOS
    available_eos = ['DH']
    out['EOS'] = str(d.get('EOS', 'none'))
    if out['EOS'] not in available_eos:
        raise IOError("Specify a valid equation of state")

    if out['EOS'] == 'DH':
        keys = ['rho0', 'P0', 'C1', 'C2']
        defaults = [877.7007, 101325, 3.5e10, 1.23]
        for k, de in zip(keys, defaults):
            out[k] = float(d.get(k, de))

    # Non-Newtonian behvior
    # ...

    print_dict(out)

    return out


def sanitize_numerics(d):

    out = {}

    out['tol'] = float(d.get('tol', 1e-6))
    out['max_it'] = int(d.get('max_it', 1000))
    out['dt'] = float(d.get('dt', 3e-10))
    out['adaptive'] = bool(d.get('adaptive', False))
    out['CFL'] = float(d.get('CFL', 0.5))

    print_dict(out)

    return out


def sanitize_gp(d):

    out = {}

    print_dict(out)

    return out
