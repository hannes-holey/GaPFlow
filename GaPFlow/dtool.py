import os
import dtoolcore
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from getpass import getuser
from ruamel.yaml import YAML
from dtool_lookup_api import query
from socket import gethostname
import yaml


from GaPFlow.utils import progressbar

try:
    import lammps
except ImportError:
    from unittest.mock import Mock as lammps
    lammps.__version__ = "0.0.0"


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

    readme_list = [yaml.full_load(ds.get_readme_content()) for ds in remote_ds]

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

    readme_list = [yaml.full_load(ds.get_readme_content())
                   for ds in dtoolcore.iter_datasets_in_base_uri(local_path)]

    print(f"Loading {len(readme_list)} local datasets in '{local_path}'.")
    for ds in dtoolcore.iter_datasets_in_base_uri(local_path):
        print(f'- {ds.uuid} ({ds.name})')

    return readme_list


def init_dataset(base_uri, suffix):

    ds_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_dataset-{suffix:03}'
    proto_ds = dtoolcore.create_proto_dataset(name=ds_name, base_uri=base_uri)
    proto_ds_path = proto_ds.uri.removeprefix('file://' + gethostname())

    # proto_ds.update_name(f"{suffix:03}_{proto_ds.uuid}")
    # new_name = os.path.join(base_uri, proto_ds.name)
    # import shutil
    # shutil.move(old_name, new_name)

    # proto_ds_path = os.path.join(base_uri, ds_name)
    # proto_ds_datapath = os.path.join(base_uri, ds_name, 'data')

    return proto_ds, proto_ds_path


def write_readme(path, Xnew, Ynew, Yerrnew, params=None):
    """Write dtool README.yml

    Parameters
    ----------
    path : str
        dtool proto dataset path
    Xnew : numpy.ndarray
        New input data
    Ynew : numpy.ndarray
        New output data
    Yerrnew : numpy.ndarray
        New output data error (signal noise)
    """

    # TODO from file
    readme_template = """
    project: Multiscale Simulation of Lubrication
    description: Automatically generated MD run of confined fluid for multiscale simulations
    owners:
      - name: Hannes Holey
        email: hannes.holey@unimi.it
        username: hannes
        orcid: 0000-0002-4547-8791
    funders:
      - organization: Deutsche Forschungsgemeinschaft (DFG)
        program: Graduiertenkolleg
        code: GRK 2450
    creation_date: {DATE}
    expiration_date: {EXPIRATION_DATE}
    software_packages:
      - name: LAMMPS
        version: {version}
        website: https://lammps.sandia.gov/
        repository: https://github.com/lammps/lammps
    """

    yaml = YAML()
    yaml.explicit_start = True
    yaml.indent(mapping=4, sequence=4, offset=2)
    metadata = yaml.load(readme_template)

    # Update metadata
    metadata["owners"][0].update(dict(username=getuser()))
    metadata["creation_date"] = date.today()
    metadata["expiration_date"] = metadata["creation_date"] + relativedelta(years=10)
    metadata["software_packages"][0]["version"] = str(lammps.__version__)

    if params is not None:
        metadata['parameters'] = {k: v for k, v in params.items()}

    out_fname = os.path.join(path, 'README.yml')

    X = [float(item) for item in Xnew]
    Y = [float(item) for item in Ynew]
    Yerr = [float(item) for item in Yerrnew]

    metadata['X'] = X
    metadata['Y'] = Y
    metadata['Yerr'] = Yerr

    with open(out_fname, 'w') as outfile:
        yaml.dump(metadata, outfile)
