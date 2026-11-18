
import yaml


def apply_overrides(config, overrides):
    """Set keys in a nested config dict by full dotted path, e.g.
    'numerics.dt' or 'implicit_solver.stabilization.oss'. Missing sections
    and keys along the path are created.
    """
    for key, value in overrides.items():
        *parents, leaf = key.split('.')
        node = config
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = value
    return config


def build_config(template, **overrides):
    config = yaml.safe_load(template)
    apply_overrides(config, overrides)
    return yaml.safe_dump(config)