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
import os
import dtoolcore
from ruamel.yaml import YAML
from dtool_lookup_api import query

from GaPFlow.utils import progressbar

yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=4, sequence=4, offset=2)


def get_readme_list_remote():
    """Get list of dtool README files for existing MD runs
    from a remote data server (via dtool_lookup_api)

    In the future, one should be able to pass a valid MongoDB
    query string to select data.

    Returns
    -------
    list
        List of dicts containing the readme content
    """

    # TODO: Pass a textfile w/ uuids or yaml with query string
    query_dict = {"readme.description": {"$regex": "Dummy"}}
    remote_ds_list = query(query_dict)

    remote_ds = [dtoolcore.DataSet.from_uri(ds['uri'])
                 for ds in progressbar(remote_ds_list,
                                       prefix="Loading remote datasets based on dtool query: ")]

    readme_list = [yaml.load(ds.get_readme_content()) for ds in remote_ds]

    return readme_list


def get_readme_list_local(local_path):
    """Get list of dtool README files for existing MD runs
    from a local directory.

    Returns
    -------
    list
        List of dicts containing the readme content
    """

    if not os.path.exists(local_path):
        os.makedirs(local_path)
        return []

    readme_list = [yaml.load(ds.get_readme_content())
                   for ds in dtoolcore.iter_datasets_in_base_uri(local_path)]

    print(f"Loading {len(readme_list)} local datasets in '{local_path}'.")
    for ds in dtoolcore.iter_datasets_in_base_uri(local_path):
        print(f'- {ds.uuid} ({ds.name})')

    return readme_list
