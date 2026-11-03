# Config library

Input files (YAML) for GaPFlow simulations, used by the tests, the tutorials and as examples.

| Folder | Contents |
|-|-|
| `validation/` | Benchmark cases with analytical or literature reference solutions. Used by the tests (`tests/test_fem_analytic.py`) and the implicit solver tutorials. |
| `basic/` | Examples with built-in constitutive laws (no GP surrogate). |
| `gp/` | Examples with Gaussian process surrogate models. Files ending in `_lammps` generate training data with LAMMPS. |
| `tutorials/` | Input files for specific tutorials. |

Run a case from the repository root, e.g.

```bash
python -m GaPFlow -i config_lib/basic/journal_1d_dowson-higginson.yaml
```

Paths in the `md` section of the `_lammps` configs point to `lammps_templates/` and are resolved relative to the current working directory, so these cases have to be started from the repository root.
