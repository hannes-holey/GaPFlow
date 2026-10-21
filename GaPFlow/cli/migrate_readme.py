#
# Copyright 2026 Hannes Holey
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
"""Migrate dtool README.yml files to the current GaPFlow schema_version.

Training datasets are dtool datasets whose README.yml carries a
'schema_version' tag (see GaPFlow.db.README_SCHEMA_VERSION). Databases
refuse to load READMEs whose version does not match; run this script on the
dataset's dtool base URI to bring them up to date first.

Each migration step is a function taking (rm: dict, **kwargs) -> dict, keyed
by the schema_version it upgrades *from*. Add new steps to _MIGRATIONS when
README_SCHEMA_VERSION is bumped.
"""
import io
from argparse import ArgumentParser

import dtoolcore
from ruamel.yaml import YAML

from ..db import README_SCHEMA_VERSION
from ..logging import get_logger

logger = get_logger("gapflow.cli.migrate_readme")

yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=4, sequence=4, offset=2)


def _migrate_v0_to_v1(rm: dict, U: float | None = None, V: float | None = None, **kwargs) -> dict:
    """Pre-wall-velocity layout -> wall velocities as base features.

    Version 0 stored X as 6 base features (rho, jx, jy, h, dhdx, dhdy)
    directly followed by extra features. Wall velocities U, V were a
    simulation-wide constant, never written to the README, so they must be
    supplied explicitly.
    """
    if U is None or V is None:
        raise ValueError(
            "Migrating a schema_version 0 README requires --U and --V (the "
            "wall velocities used to generate this dataset)."
        )

    rm['X'] = rm['X'][:6] + [U, V] + rm['X'][6:]
    rm['schema_version'] = 1

    return rm


# Registry of migration steps, keyed by the schema_version they upgrade from.
_MIGRATIONS = {
    0: _migrate_v0_to_v1,
}


def migrate_readme(rm: dict, **kwargs) -> tuple[dict, bool]:
    """Migrate a single README dict to README_SCHEMA_VERSION.

    Parameters
    ----------
    rm : dict
        README content, as returned by ``yaml.load``.
    **kwargs
        Forwarded to each migration step (e.g. U, V for the v0 -> v1 step).

    Returns
    -------
    tuple[dict, bool]
        The (possibly modified) README dict, and whether it was changed.
    """
    version = rm.get('schema_version', 0)

    if version > README_SCHEMA_VERSION:
        raise ValueError(
            f"README has schema_version {version}, newer than "
            f"{README_SCHEMA_VERSION} known to this installation. Update GaPFlow first."
        )

    changed = version < README_SCHEMA_VERSION

    while version < README_SCHEMA_VERSION:
        step = _MIGRATIONS.get(version)
        if step is None:
            raise ValueError(f"No migration step registered for schema_version {version}.")
        rm = step(rm, **kwargs)
        version = rm['schema_version']

    return rm, changed


def get_parser():

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('path', help="Local dtool base URI (directory containing dtool datasets).")
    parser.add_argument('--U', type=float, default=None,
                        help="Wall velocity U, required to migrate schema_version 0 READMEs.")
    parser.add_argument('--V', type=float, default=None,
                        help="Wall velocity V, required to migrate schema_version 0 READMEs.")
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help="Report what would change without writing anything.")

    return parser


def main():

    args = get_parser().parse_args()

    n_migrated = 0
    n_skipped = 0

    for ds in dtoolcore.iter_datasets_in_base_uri(args.path):
        rm = yaml.load(ds.get_readme_content())
        version_before = rm.get('schema_version', 0)

        rm, changed = migrate_readme(rm, U=args.U, V=args.V)

        if not changed:
            n_skipped += 1
            continue

        logger.info("%s (%s): schema_version %d -> %d",
                    ds.name, ds.uuid, version_before, rm['schema_version'])

        if not args.dry_run:
            buf = io.StringIO()
            yaml.dump(rm, buf)
            ds.put_readme(buf.getvalue())

        n_migrated += 1

    action = "Would migrate" if args.dry_run else "Migrated"
    logger.info("%s %d dataset(s), %d already up to date.", action, n_migrated, n_skipped)
