#
# Copyright 2024-2025 Hannes Holey
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
import numpy as np
import dtoolcore
from socket import gethostname
try:
    import lammps
except ImportError:
    pass
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from getpass import getuser
from ruamel.yaml import YAML
from dtool_lookup_api import query
from scipy.stats import qmc
import scipy.constants as sci


from hans.tools import progressbar, bordered_text
from hans.multiscale.md_setup import write_template, build_template
from hans.multiscale.md import run, mpirun
from hans.multiscale.util import variance_of_mean


class Database:
    """
    Holds the MD training data and controls generation of new MD runs.
    """

    def __init__(self, gp, md, const_fields, eos_func, tau_func):
        """

        Parameters
        ----------
        gp : dict
            GP configuration
        md : dict
            MD configuration
        const_fields : dict
            Holds constant fields which can be used as features for the GP
        eos_func : callable
            Computes the EoS if data does not come from MD
        tau_func : callable
            Computes the viscous stress if data does not come from MD
        """

        self.eos_func = eos_func
        self.tau_func = tau_func
        self.gp = gp
        self.md = md

        self.system = self.md.get('system', 'lj')
        self.debug = bool(self.md.get('debug', False))

        if self.system == "lj":
            self._stress_scale = 1.
            self._var_stress_scale = 1.
        else:  # moltemplate
            # Alkane/gold
            # from kcal/mol/A^3 to g/mol/A/fs^2
            self._stress_scale = float(self.md.get('scaleRM', sci.calorie * 1e-4))
            self._var_stress_scale = self._stress_scale**2

        # Constant input:
        # - gap (default)
        # - slip
        # - gap_angle
        self.mode = self.gp.get('mode', 'gap')

        if self.mode == "slip":
            self.c = const_fields['kappa']
            self._features = ["slip_param", "density", "mass_flux_x", "mass_flux_y"]

        elif self.mode == "gap_angle":
            self.c = np.vstack([
                const_fields['gap_height'][0][None, :, :],
                np.arctan(const_fields['gap_height'][1][None, :, :]) / np.pi * 180.
            ])
            self._features = ["gap_height", "angle", "density", "mass_flux_x", "mass_flux_y"]

        else:  # mode == "gap""
            self.c = const_fields['gap_height'][0][None, :, :]
            self._features = ["gap_height", "density", "mass_flux_x", "mass_flux_y"]

        self.h = const_fields['gap_height']

        self._init_database(len(self._features), 13)

    def __del__(self):
        self.save()

    @property
    def num_features(self):
        return len(self._features)

    @property
    def size(self):
        return self.Xtrain.shape[1]

    def save(self):
        np.save(f'Ytrain-{self.size:03d}.npy', self.Ytrain)
        np.save(f'Xtrain-{self.size:03d}.npy', self.Xtrain)
        np.save(f'Ytrainvar-{self.size:03d}.npy', self.Yerr)

    def _get_readme_list_remote(self):
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

        yaml = YAML()

        readme_list = [yaml.load(ds.get_readme_content()) for ds in remote_ds]

        return readme_list

    def _get_readme_list_local(self):
        """Get list of dtool README files for existing MD runs
        from a local directory.

        Returns
        -------
        list
            List of dicts containing the readme content
        """

        if not os.path.exists(self.gp['local']):
            os.makedirs(self.gp['local'])
            return []

        readme_paths = [os.path.join(ds.uri.removeprefix(f'file://{gethostname()}'), 'README.yml')
                        for ds in dtoolcore.iter_datasets_in_base_uri(self.gp['local'])]

        yaml = YAML()

        readme_list = []
        for readme in readme_paths:
            with open(readme, 'r') as instream:
                readme_list.append(yaml.load(instream))

        print(f"Loading {len(readme_list)} local datasets in '{self.gp['local']}'.")

        return readme_list

    def _init_database(self, input_dim, output_dim):
        """Initialize the buffers that store the training data with results from previous MD runs.
        If no runs are available, empty buffers are created.

        Parameters
        ----------
        input_dim : int
            Number of input dimensions (features)
        output_dim : int
            Number of output dimensions
        """

        if bool(self.gp['remote']):
            readme_list = self._get_readme_list_remote()
        else:
            readme_list = self._get_readme_list_local()

        if len(readme_list) > 0:

            Xtrain = []
            Ytrain = []
            Yerr = []

            for rm in readme_list:
                X = np.array(rm['X'])
                Y = np.array(rm['Y'])

                # older readme files have the height gradients in X which is never used for training
                # exclude these entries.
                if len(rm['X']) == 6:
                    X = X[[0, 3, 4, 5]]

                Xtrain.append(X)
                Ytrain.append(Y)

                if 'Yerr' in rm.keys():
                    Yerr.append(rm['Yerr'])
                else:
                    Yerr.append([0., 0., 0.])

            self.Xtrain = np.array(Xtrain).T
            # REMINDER: Need to change earlier READMEs with wrong units
            # This assumes that READMEs are correctly written
            self.Ytrain = np.array(Ytrain).T  # * self._stress_scale
            self.Yerr = np.array(Yerr).T  # * self._var_stress_scale

            self.save()

        else:
            print("No matching dtool datasets found. Start with empty database.")
            self.Xtrain = np.empty((input_dim, 0))
            self.Ytrain = np.empty((output_dim, 0))
            self.Yerr = np.empty((3, 0))

    def initialize(self, Q, Ninit, sampling='lhc'):
        """Build initial database with different sampling strategies.

        Parameters
        ----------
        Q : np.ndarray
            Current solution (initial conditions in most cases)
        Ninit : int
            Number of initial datapoints
        sampling: str
            (Quasi-)random sampling strategy. Select one of Latin hypercube ('lhc'),
            Sobol ('sobol') or random ('random') sampling. The default is 'lhc'.
        """

        Nsample = Ninit - self.size

        # Bounds for quasi random sampling of initial database
        jabs = np.hypot(np.mean(Q[1, :]), np.mean(Q[2, :]))
        rho = np.mean(Q[0, :])

        l_bounds = np.amin(self.c, axis=(1, 2))
        u_bounds = np.amax(self.c, axis=(1, 2))

        l_bounds[1] = -1e-10
        u_bounds[1] = 1e-10

        l_bounds = np.hstack([l_bounds, [0.99 * rho, 0.5 * jabs, 0.]])
        u_bounds = np.hstack([u_bounds, [1.01 * rho, 1.5 * jabs, 0.5 * jabs]])

        dim = len(l_bounds)

        # Sampling
        if sampling == 'random':
            scaled_samples = l_bounds + np.random.random_sample([Nsample, dim]) * (u_bounds - l_bounds)

        elif sampling == 'lhc':
            sampler = qmc.LatinHypercube(d=dim)
            sample = sampler.random(n=Nsample)
            scaled_samples = qmc.scale(sample, l_bounds, u_bounds)

        elif sampling == 'sobol':
            sampler = qmc.Sobol(d=dim)
            m = int(np.log2(Nsample))
            if int(2**m) != Nsample:
                m = int(np.ceil(np.log2(Nsample)))
                print(f'Sample size should be a power of 2 for Sobol sampling. Use Ninit={2**m}.')
            sample = sampler.random_base2(m=m)
            scaled_samples = qmc.scale(sample, l_bounds, u_bounds)

        c_init = scaled_samples[:, :self.c.shape[0]]
        rho_init = scaled_samples[:, -3]
        jx_init = scaled_samples[:, -2]
        jy_init = scaled_samples[:, -1]

        if Q.shape[2] <= 3:
            jy_init = np.zeros_like(jx_init)

        # Assemble
        if self.mode == "slip":
            # The slip version only works with constant height (so far)
            h_init = np.ones_like(c_init) * np.mean(self.h[0, :, 1])
            h_gradx_init = np.zeros_like(h_init)
            h_grady_init = np.zeros_like(h_init)
        else:  # mode == "gap"
            h_init = c_init
            h_gradx_init = np.ones_like(h_init) * np.mean(self.h[1, :, 1])
            h_grady_init = np.ones_like(h_init) * np.mean(self.h[2, :, 1])

        Hnew = np.vstack([h_init,
                          h_gradx_init,
                          h_grady_init
                          ])

        # TODO: should work for all shapes of c_init
        # Cnew = np.vstack([c_init,
        #                   ])
        Cnew = c_init.T
        Qnew = np.vstack([rho_init, jx_init, jy_init])
        Xnew = np.vstack([Cnew, Qnew])

        self.Xtrain = np.hstack([self.Xtrain, Xnew])

        self._update_outputs(Xnew, Hnew)

    def update(self, Q, ix, iy):
        """Update the database.

        Parameters
        ----------
        Q : numpy.ndarray
            Current solution at (ix, iy)
        ix : int
            Location on the 2D Cartesian grid (x index)
        iy : int
            Location on the 2D Cartesian grid (y index)
        """
        Xnew = self._update_inputs(Q, ix, iy)
        Hnew = self.get_gap_height(ix, iy)

        self._update_outputs(Xnew, Hnew)

    def _update_inputs(self, Qnew, ix, iy):
        """Fill training dataset with new inputs.

        Parameters
        ----------
        Qnew : numpy.ndarray
            Current solution at (ix, iy)
        ix : int
            Location on the 2D Cartesian grid (x index)
        iy : int
            Location on the 2D Cartesian grid (y index)

        Returns
        -------
        numpy.ndarray
            New training data
        """
        Cnew = self.get_constant(ix, iy)
        Xnew = np.vstack([Cnew, Qnew])
        self.Xtrain = np.hstack([self.Xtrain, Xnew])

        return Xnew

    def _update_outputs(self, Xnew, Hnew):
        """Fill training dataset with new output.

        Parameters
        ----------
        Xnew : numpy.ndarray
            New inputs
        Hnew : numpy.ndarray
            Gap hieght at new input locations
        """

        Ynew = np.zeros((13, Xnew.shape[1]))
        Yerrnew = np.zeros((3, Xnew.shape[1]))

        # In current version of heteroscedastic multi-output GP (single kernel):
        # one error for normal and one for shear stress.
        # Otherwise need individual kernels for tau_xy and tau_xy,
        # and possibly correlations between outputs.

        for i in range(Xnew.shape[1]):

            num = self.Ytrain.shape[1] + i + 1
            base_uri = self.gp['local']
            ds_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_dataset-{num:03}'
            proto_ds = dtoolcore.create_proto_dataset(name=ds_name, base_uri=base_uri)
            proto_datapath = os.path.join(base_uri, ds_name, 'data')

            data_acq = 'MD simulation' if self.md is not None else '"MD simulation"'

            text = [f'Run next {data_acq} in: {proto_datapath}']
            text.append('---')

            for j, (X, f) in enumerate(zip(Xnew[:, i], self._features)):
                text.append(f'Input {j+1}: {X:8.5f} ({f})')

            print(bordered_text('\n'.join(text)))

            skip_md = self.md is None or self.debug

            # Run LAMMPS...
            if self.md is not None:

                # Store directory
                basedir = os.getcwd()

                # Run inside dtool dataset
                os.chdir(proto_datapath)

                # Run MD with fixed number of cores in proto dataset
                nworker = self.write_input_files(proto_ds, Xnew, Hnew, i)

                if not skip_md:

                    if nworker > 1:
                        mpirun(self.system, nworker)
                    else:
                        run(self.system)

                    # Get stress
                    # Apply unit conversion from LAMMPS output to READMEs
                    md_data = np.loadtxt('stress_wall.dat') * self._stress_scale

                    if md_data.shape[1] == 5:
                        # 1D
                        # step, pL_t, tauL_t, pU_t, tauU_t = np.loadtxt('stress_wall.dat', unpack=True)

                        # timeseries
                        pressL_t = md_data[:, 1]
                        pressU_t = md_data[:, 3]
                        tauL_t = md_data[:, 2]
                        tauU_t = md_data[:, 4]

                        # mean
                        pressL = np.mean(pressL_t)
                        pressU = np.mean(pressU_t)
                        tauL = np.mean(tauL_t)
                        tauU = np.mean(tauU_t)

                        # variance of mean
                        pL_err = variance_of_mean(pressL_t)
                        pU_err = variance_of_mean(pressU_t)
                        tauxzL_err = variance_of_mean(tauL_t)
                        tauxzU_err = variance_of_mean(tauU_t)

                        # fill into buffer
                        Ynew[0, i] = (pressL + pressU) / 2.
                        Ynew[5, i] = tauL
                        Ynew[11, i] = tauU
                        Yerrnew[0, i] = (pL_err + pU_err) / 2.
                        Yerrnew[1, i] = (tauxzL_err + tauxzU_err) / 2.

                    elif md_data.shape[1] == 7:
                        # 2D
                        # step, pL_t, tauxzL_t, pU_t, tauxzU_t, tauyzL_t, tauyzU_t = np.loadtxt('stress_wall.dat', unpack=True)

                        # timeseries data
                        pressL_t = md_data[:, 1]
                        pressU_t = md_data[:, 3]
                        tauxzL_t = md_data[:, 2]
                        tauxzU_t = md_data[:, 4]
                        tauyzL_t = md_data[:, 5]
                        tauyzU_t = md_data[:, 6]

                        # mean
                        pressL = np.mean(pressL_t)
                        pressU = np.mean(pressU_t)
                        tauxzL = np.mean(tauxzL_t)
                        tauxzU = np.mean(tauxzU_t)
                        tauyzL = np.mean(tauyzL_t)
                        tauyzU = np.mean(tauyzU_t)

                        # variance of mean
                        pL_err = variance_of_mean(pressL_t)
                        pU_err = variance_of_mean(pressU_t)
                        tauxzL_err = variance_of_mean(tauxzL_t)
                        tauxzU_err = variance_of_mean(tauxzU_t)
                        tauyzL_err = variance_of_mean(tauyzL_t)
                        tauyzU_err = variance_of_mean(tauyzU_t)

                        # fill into buffer
                        Ynew[0, i] = (pressL + pressU) / 2.
                        Ynew[4, i] = tauyzL
                        Ynew[5, i] = tauxzL
                        Ynew[10, i] = tauyzU
                        Ynew[11, i] = tauxzU
                        Yerrnew[0, i] = (pL_err + pU_err) / 2.
                        Yerrnew[1, i] = (tauxzL_err + tauxzU_err) / 2.
                        Yerrnew[2, i] = (tauyzL_err + tauyzU_err) / 2.

            os.chdir(basedir)

            # ... or use a (possibly noisy) constitutive law
            if skip_md:
                # Artificial noise
                pnoise = np.random.normal(0., np.sqrt(self.gp['noisePress']), size=(1, Xnew.shape[1]))
                snoise = np.random.normal(0., np.sqrt(self.gp['noiseShear']), size=(1, Xnew.shape[1]))

                # pressure
                Ynew[0, i] = self.eos_func(Xnew[1, i]) + pnoise[:, i]

                if self.mode == "slip":
                    tau = self.tau_func(Xnew[1:, i, None], Hnew[:, i, None], Xnew[0, i, None][0])
                else:
                    tau = self.tau_func(Xnew[1:, i, None], Hnew[:, i, None], 0.)

                Ynew[1:, i, None] = tau + snoise[:, i, None]

                Yerrnew = np.zeros((3, Xnew.shape[1]))
                Yerrnew[0, i] = self.gp['noisePress']
                Yerrnew[1:, i] = self.gp['noiseShear']

            self.write_readme(os.path.join(base_uri, ds_name), Xnew[:, i], Ynew[:, i], Yerrnew[:, i])

            proto_ds.freeze()

            if self.gp['remote']:
                dtoolcore.copy(proto_ds.uri, self.gp['storage'])

        self.Ytrain = np.hstack([self.Ytrain, Ynew])
        self.Yerr = np.hstack([self.Yerr, Yerrnew])

        self.save()

    def write_input_files(self, proto_ds, Xnew, Hnew, index):

        if self.system == "lj":

            num_cpu = self.md['ncpu']

            # Move inputfiles to proto dataset
            proto_ds.put_item(self.md['wallfile'], 'in.wall')
            proto_ds.put_item(self.md['infile'], 'in.run')

            # write variables file (# FIXME: height, indices not hardcoded)
            var_str = f'variable input_gap equal {Hnew[0, index]}\n' + \
                f'variable input_dens equal {Xnew[-3, index]}\n' + \
                f'variable input_fluxX equal {Xnew[-2, index]}\n' + \
                f'variable input_fluxY equal {Xnew[-1, index]}\n'

            if self.mode == "slip":
                var_str += f'variable input_kappa equal {Xnew[0, index]}\n'

            excluded = ['infile', 'wallfile', 'ncpu']
            for k, v in self.md.items():
                if k not in excluded:
                    var_str += f'variable {k} equal {v}\n'

            with open(os.path.join('in.param'), 'w') as f:
                f.writelines(var_str)

        else:  # moltemplate (alkane/gold)
            args = self.md

            # Move inputfiles to proto dataset
            os.makedirs(os.path.join("moltemplate_files"))
            os.makedirs(os.path.join("static"))

            proto_ds.put_item(self.md['fftemplate'],
                              os.path.join("moltemplate_files", os.path.basename(self.md['fftemplate'])))
            proto_ds.put_item(self.md['topo'],
                              os.path.join("moltemplate_files", os.path.basename(self.md['topo'])))

            for f in os.listdir(self.md["staticFiles"]):
                proto_ds.put_item(os.path.join(self.md["staticFiles"], f), os.path.join("static", f))

            # TODO: separate section of metadata

            args["density"] = float(Xnew[-3, index])
            args["fluxX"] = float(Xnew[-2, index])
            args["fluxY"] = float(Xnew[-1, index])

            if self.mode.startswith('gap'):
                args["gap_height"] = float(Xnew[0, index])

            if self.mode == "gap_angle":
                args["rotation"] = float(Xnew[1, index])

            num_cpu = write_template(args)
            build_template(args)

        return num_cpu

    def write_readme(self, path, Xnew, Ynew, Yerrnew):
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

        if self.md is not None:
            # metadata['parameters'] = {k: self.md[k] for k in ['cutoff', 'temp', 'vWall', 'tsample']}
            metadata['parameters'] = {k: v for k, v in self.md.items()}

        out_fname = os.path.join(path, 'README.yml')

        X = [float(item) for item in Xnew]
        Y = [float(item) for item in Ynew]
        Yerr = [float(item) for item in Yerrnew]

        metadata['X'] = X
        metadata['Y'] = Y
        metadata['Yerr'] = Yerr

        with open(out_fname, 'w') as outfile:
            yaml.dump(metadata, outfile)

    def get_gap_height(self, ix, iy=1):
        """Get the gap height at the input location.

        Parameters
        ----------
        ix : int
            Location on the 2D Cartesian grid (x index)
        iy : int, optional
            Location on the 2D Cartesian grid (y index)
            (the default is 1, which [default_description])

        Returns
        -------
        numpy.ndarray
            Local gap height (and gradients)
        """

        Hnew = self.h[:, ix, iy]
        if Hnew.ndim == 1:
            Hnew = Hnew[:, None]

        return Hnew

    def get_constant(self, ix, iy=1):
        """Get the constant field at the input location.
        Currently, either gap height, or a variable quantifying wall slip.

        Parameters
        ----------
        ix : int
            Location on the 2D Cartesian grid (x index)
        iy : int, optional
            Location on the 2D Cartesian grid (y index)
            (the default is 1, which [default_description])

        Returns
        -------
        numpy.ndarray
            Local constant field
        """

        Cnew = self.c[:, ix, iy]
        if Cnew.ndim == 1:
            Cnew = Cnew[:, None]

        return Cnew
