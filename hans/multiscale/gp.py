#
# Copyright 2023, 2025 Hannes Holey
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

import abc
from GPy.models import GPRegression, GPHeteroscedasticRegression
from GPy.kern import Matern32
import numpy as np
import os
from copy import deepcopy
from scipy.stats import qmc


class GaussianProcess:
    """Abstract base class for all Gaussian process surrogates.

    Attributes
    ----------
    name : str
        Name of the surrogate model:
        - shear: Wall shear stress (xz) in 1D simulations
        - shearXZ: Wall shear stress (xz) in 2D simulations
        - shearYZ: Wall shear stress (yz) in 2D simulations
        - press: Wall normal stress (pressure) in 1/2D simulations

    ndim : int
        (Cartesian) dimensions (either 1 or 2)
    """

    __metaclass__ = abc.ABCMeta

    name: str
    ndim: int

    def __init__(self,
                 active_dim,
                 db,
                 gp,
                 Xmask,
                 Ymask,
                 atol,
                 noise_variance):
        """
        Constructor.

        Parameters
        ----------
        active_dim : int
            Number of active input dimensions.
        db : hans.multiscale.db.Database
            Attached training database
        gp : dict
            GP configuration
        Xmask : list
            Input mask
        Ymask : list
            Output mask
        atol : float
            Absolute tolerance
        noise_variance : float
            Noise variance parameter
        """

        self.active_dim = active_dim
        self.db = db
        self.Xmask = Xmask
        self.Ymask = Ymask

        _Yerr_map = {'press': 0, 'shear': 1, 'shearXZ': 1, 'shearYZ': 2}
        self.index = _Yerr_map[self.name]

        # Kernel
        self.kern = Matern32(active_dim, ARD=True)

        # Noise
        self.heteroscedastic_noise = gp['heteroscedastic']
        self.noise_fixed = gp['noiseFixed']
        self.noise_variance = noise_variance

        # Tolerance
        self.atol = atol
        self.rtol = gp['rtol']
        self.tol = 0.
        self.use_atol = True

        # Common parameters
        self.options = gp
        self.step = 0
        self.maxvar = np.inf
        self._reset = False

        self._init_outfile()

    def __del__(self):

        self._fit()
        self.model.save_model(f'gp_{self.name}.json', compress=True)
        self._write_history()
        self.file.close()

    @property
    def dbsize(self):
        return self.db.size

    @property
    def trusted(self):
        return self.maxvar < self.tol

    @property
    def reset(self):
        return self._reset

    def reset_reset(self):
        """
        Set reset parameter back to False
        """
        self._reset = False

    def increment(self):
        """
        Increment internal counter by 1.
        """
        self.step += 1

    def setup(self, q):
        """Initial setup of the GP model.
        Triggers MD runs, if not enough training data is there.


        Parameters
        ----------
        q : numpy.ndarray
            Current solution
        """
        nx, ny = q.shape[1:]
        self.rshape = (nx, ny if ny > 3 else 1)

        self._set_solution(q)

        if self.dbsize < self.options['Ninit']:
            self.db.initialize(self.sol,
                               self.options['Ninit'],
                               self.options['sampling'])

        self.last_fit_dbsize = self.dbsize
        self._fit()
        Xtest = self._get_test_input()
        self._raw_predict(Xtest)
        self._write_history()

    def _raw_predict(self, Xtest):
        """Make prediction for the given test locations.

        Parameters
        ----------
        Xtest : numpy.ndarray
            Test data

        Returns
        -------
        numpy.ndarray
            GP posterior mean
        numpy.ndarray
            GP posterior variance
        """

        mean, cov = self.model.predict_noiseless(Xtest / self.Xnorm, full_cov=False)
        mean *= self.Ynorm
        cov *= self.Ynorm**2
        self.maxvar = np.amax(cov)

        # absolute tolerance is multiple of mean noise variance from data
        atol = self.atol * np.sqrt(self.model.Gaussian_noise.variance[0]) * self.Ynorm
        rtol = self.rtol * (np.amax(mean) - np.amin(mean))

        self.tol = max(atol, rtol)**2
        self.use_atol = atol > rtol

        if self.ndim == 1:
            return mean.T[:, :, np.newaxis], cov
        else:
            return np.transpose(mean.reshape(*self.rshape, -1), (2, 0, 1)), cov

    def predict(self, q, in_predictor):
        """Wrapper around the raw prediction and hyperparameter optimization.
        This method is called from the time integration step.
        Triggers active learning if uncertainty tolerance is not met.


        Parameters
        ----------
        q : numpy.ndarray
            Current solution
        in_predictor : bool
            Indicates whether the call is from the predictor (True) or the corrector step (False)

        Returns
        -------
        Returns
        -------
        numpy.ndarray
            GP posterior mean
        numpy.ndarray
            GP posterior variance
        """

        self._set_solution(q)
        Xtest = self._get_test_input()
        old_maxvar = deepcopy(self.maxvar)

        if in_predictor:
            # Predictor step: Refit after database has grown
            if self.last_fit_dbsize != self.dbsize:
                self.gp_print()
                self._fit()
                mean, var = self._raw_predict(Xtest)
                self._write_history()
                self.gp_print(f'>>>> {"Reason":<18}: {"DB changed"}')
                self.gp_print(f'>>>> {"Max. var. (old)":<18}: {old_maxvar:.3e}')
                self.gp_print(f'>>>> {"Max. var. (new)":<18}: {self.maxvar:.3e}')
                self.gp_print(f'>>>> {"Tolerance":<18}: {self.tol:.3e} ({'A' if self.use_atol else 'R'})')
                suffix = ' (waiting)' if self._reset else ''
                self.gp_print(f'>>>> {"Accepted":<18}: {self.trusted} {suffix}')
            else:
                mean, var = self._raw_predict(Xtest)
        else:
            # Corrector step: Use same state of the model as in predictor
            mean, var = self._raw_predict(Xtest)

        if in_predictor and not self._reset:
            counter = 0
            while (not self.trusted) and (counter < self.options['maxSteps']):
                found_newX = self._active_learning_step(mean, var)
                counter += 1
                old_maxvar = deepcopy(self.maxvar)
                if found_newX:
                    self.gp_print()
                    self._fit()
                    mean, var = self._raw_predict(Xtest)
                    self._write_history()
                    self.gp_print(f'>>>> {"Reason":<18}: AL {counter}/{self.options["maxSteps"]}')
                    self.gp_print(f'>>>> {"Max. var. (old)":<18}: {old_maxvar:.3e}')
                    self.gp_print(f'>>>> {"Max. var. (new)":<18}: {self.maxvar:.3e}')
                    self.gp_print(f'>>>> {"Tolerance":<18}: {self.tol:.3e} ({'A' if self.use_atol else 'R'})')
                    self.gp_print(f'>>>> {"Accepted":<18}: {self.trusted}')
                else:
                    self.gp_print(f'>>>> No data added ({counter}/{self.options["maxSteps"]})')

            if not self.trusted:
                self._reset = True

        return mean, var

    def predict_gradient(self):
        """Predict the gradient of the GP w.r.t. the inputs.
        Used to estimate the speed of sound from the EoS.

        Returns
        -------
        numpy.ndarray
            Predictive gradient
        numpy.ndarray
            Variance of the predictive gradient
        """
        Xtest = self._get_test_input()
        dq_dX, dv_dX = self.model.predictive_gradients(Xtest / self.Xnorm)
        dq_dX *= self.Ynorm / self.Xnorm.T[None, :, :]

        if self.ndim == 1:
            return dq_dX.T[0], dv_dX
        else:
            return np.transpose(dq_dX.reshape(*self.rshape, -1), (2, 0, 1)), dv_dX

    def _fit(self):
        """
        Hyperparameter optimization.
        """

        self._build_model()

        num_restarts = self.options['optRestarts']

        v0, l0 = self._initial_guess_kernel_params(2)
        self.kern.lengthscale = l0
        self.kern.variance = v0

        # Constrain lengthscales
        Xdelta = np.maximum(self.model.X.max(0) - self.model.X.min(0), 1e-8)

        for i in range(self.active_dim):
            Xlimits = [Xdelta[i] / self.model.X.shape[0], 100 * Xdelta[i]]
            # assert l0[i] < Xlimits[1]
            # assert l0[i] > Xlimits[0]
            self.kern.lengthscale[[i]].constrain_bounded(*Xlimits, warning=False)

        self.model.optimize()
        # self.model.optimize_restarts(verbose=False)
        best_model = self.model.copy()
        best_mll = np.copy(-self.model.log_likelihood())

        for i in range(num_restarts):
            # Randomize initial hyperparameters
            skip_update = False
            self.kern.lengthscale = l0 * np.exp(np.random.normal(size=self.active_dim))
            self.kern.variance = v0 * np.exp(np.random.normal())
            try:
                self.model.optimize('lbfgsb', max_iters=1000)
                # self.model.optimize_restarts(verbose=False)
            except np.linalg.LinAlgError:
                skip_update = True

            if not skip_update:
                current_mll = -self.model.log_likelihood()

                if current_mll < best_mll:
                    best_model = self.model.copy()
                    best_mll = np.copy(current_mll)

            to_stdout = f'>>>> GP_{self.name} {self.step + 1:8d} | DB size {self.dbsize:3d} | '
            to_stdout += f'Fit {i+1:2d}/{num_restarts}: NMLL (current, best): {current_mll:.3f}, {best_mll:.3f}'
            self.gp_print(to_stdout, flush=True, end='\n' if i + 1 == num_restarts else '\r')

        self.last_fit_dbsize = self.dbsize

        self.model = best_model
        self.model.save_model(os.path.join(self.options['local'],
                                           f'gp_{self.name}-{self.dbsize}.json'), compress=True)

        # Also save parameters as numpy array.
        np.save(os.path.join(self.options['local'], f'gp_{self.name}-{self.dbsize}.npy'), self.model.param_array)

    def _active_learning_step(self, mean, var):
        """Make active learning step, i.e. select next training point based on max. variance.


        Parameters
        ----------
        mean : numpy.ndarray
            GP predictive mean
        var : numpy.ndarray
            GP predictive variance

        Returns
        -------
        bool
            Returns true if another training point is found that is not too similar from the exisiting ones.
        """
        success = False

        # Acquisiition function
        xnext = np.argsort(var[:, 0])[::-1]
        _, nx, ny = mean.shape

        # try to find a point not too close to previous ones
        for i, x in enumerate(xnext):
            ix, iy = np.unravel_index(x, (nx, ny))
            success = not self._similarity_check(ix, iy if self.ndim == 2 else 1)

            if success:
                # Add to training data
                self._update_database(ix, iy)
                break

        return success

    def _init_outfile(self):
        """
        Initialize output file (txt)
        """

        fname = f'gp_{self.name}.out'
        if os.path.exists(fname):
            os.remove(fname)
        self.file = open(fname, 'w', buffering=1)

        num_params = len(self.kern.param_array)
        self.file.write(f"# Gaussian process: {self.name}\n# step db_size kernel_params[{num_params}] maxvar tol nmll\n")

    def _write_history(self):
        """
        Write current status of the GP into the output file. 
        """

        per_step = [self.step, self.dbsize]
        [per_step.append(param) for param in self.kern.param_array]

        if self.heteroscedastic_noise:
            per_step.append(np.mean(self.model.het_Gauss.variance))
        else:
            per_step.append(self.model.Gaussian_noise.variance[0] * self.Ynorm**2)

        per_step.append(self.maxvar)
        per_step.append(self.tol)
        per_step.append(-self.model.log_likelihood())
        per_step.append('A' if self.use_atol else 'R')

        fmt = ["{:8d}", "{:8d}"] + (len(per_step) - 2) * ["{:8e}"] + ["{:s}"]
        per_step = [f.format(item) for f, item in zip(fmt, per_step)]
        out_str = " ".join(per_step) + '\n'

        self.file.write(out_str)

    def _build_model(self):
        """
        Initialize GP model        
        """

        self.Xnorm = np.max(np.abs(self.db.Xtrain[self.Xmask, :].T), axis=0)[None, :]
        self.Ynorm = np.max(np.abs(self.db.Ytrain[self.Ymask, :].T))

        if self.heteroscedastic_noise:
            self.model = GPHeteroscedasticRegression(self.db.Xtrain[self.Xmask, :].T / self.Xnorm,
                                                     self.db.Ytrain[self.Ymask, :].T / self.Ynorm,
                                                     self.kern)
        else:
            self.model = GPRegression(self.db.Xtrain[self.Xmask, :].T / self.Xnorm,
                                      self.db.Ytrain[self.Ymask, :].T / self.Ynorm,
                                      self.kern)
        self._set_noise()

    def _set_noise(self):

        if self.heteroscedastic_noise:
            self.model.het_Gauss.variance = (self.db.Yerr[self.index, :].T / self.Ynorm**2)[:, None]
            if self.noise_fixed:
                self.model.het_Gauss.variance.fix()
        else:
            self.model.Gaussian_noise.variance = np.mean(self.db.Yerr[self.index, :].T / self.Ynorm**2)
            if self.noise_fixed:
                self.model.Gaussian_noise.variance.fix()

    def _initial_guess_kernel_params(self, level=0):
        """Make an initial guess of the kernel hyperparameters (length scale and variance).

        Implements methods proposed in https://arxiv.org/abs/2101.09747

        Parameters
        ----------
        level : int, optional
            Select the level of initial hyperparameter tuning.
            The default is 0, which does not optimize the initial guess.
            (1 optimizes variances analytically, 2 additionally chooses best length scales from a grid search)

        Returns
        -------
        float
            Kernel variance
        numpy.ndarray
            Kernel lengthscales
        """

        l0 = np.maximum(np.std(self.model.X, axis=0), 1e-8)
        v0 = np.var(self.model.Y)

        def variance_opt(ls):
            # Optimize NMLL w.r.t. variance analytically, see https://arxiv.org/abs/2101.09747
            self.kern.lengthscale = ls
            Ntrain = self.model.X.shape[0]
            K_inv = (self.kern.variance.values[0] * self.model.posterior.woodbury_inv).copy()
            y = self.model.Y.values.copy()
            # Assuming zero mean
            v0 = np.mean(np.diag(((y).T @ K_inv @ (y) / Ntrain)))
            return v0

        if level == 1:
            # Only variance, with const, length scale
            v0 = variance_opt(l0)

        if level == 2:
            # Grid search for length scales, also see https://arxiv.org/abs/2101.09747
            min_obj = np.inf
            best = np.hstack([v0, l0])

            sampler = qmc.LatinHypercube(self.active_dim)
            sample = sampler.random(n=1000)
            scaled_samples = qmc.scale(sample, np.ones(self.active_dim) * 0.01, np.ones(self.active_dim) * 100.)

            for s in scaled_samples:
                try:
                    _v = variance_opt(s * l0)
                    self.kern.variance = _v

                    if self.model.objective_function() < min_obj:
                        min_obj = np.copy(self.model.objective_function())
                        best = np.copy(self.kern.param_array)

                except np.linalg.LinAlgError:
                    pass

            v0 = best[0]
            l0 = best[1:]

        return v0, l0

    def _fit(self):

        self._build_model()

        num_restarts = self.options['optRestarts']

        v0, l0 = self._initial_guess_kernel_params(2)
        self.kern.lengthscale = l0
        self.kern.variance = v0

        # Constrain lengthscales
        Xdelta = np.maximum(self.model.X.max(0) - self.model.X.min(0), 1e-8)

        for i in range(self.active_dim):
            Xlimits = [Xdelta[i] / self.model.X.shape[0], 100 * Xdelta[i]]
            # assert l0[i] < Xlimits[1]
            # assert l0[i] > Xlimits[0]
            self.kern.lengthscale[[i]].constrain_bounded(*Xlimits, warning=False)

        self.model.optimize()
        # self.model.optimize_restarts(verbose=False)
        best_model = self.model.copy()
        best_mll = np.copy(-self.model.log_likelihood())

        for i in range(num_restarts):
            # Randomize initial hyperparameters
            skip_update = False
            self.kern.lengthscale = l0 * np.exp(np.random.normal(size=self.active_dim))
            self.kern.variance = v0 * np.exp(np.random.normal())
            try:
                self.model.optimize('lbfgsb', max_iters=1000)
                # self.model.optimize_restarts(verbose=False)
            except np.linalg.LinAlgError:
                skip_update = True

            if not skip_update:
                current_mll = -self.model.log_likelihood()

                if current_mll < best_mll:
                    best_model = self.model.copy()
                    best_mll = np.copy(current_mll)

            to_stdout = f'>>>> GP_{self.name} {self.step + 1:8d} | DB size {self.dbsize:3d} | '
            to_stdout += f'Fit {i+1:2d}/{num_restarts}: NMLL (current, best): {current_mll:.3f}, {best_mll:.3f}'
            self.gp_print(to_stdout, flush=True, end='\n' if i + 1 == num_restarts else '\r')

        self.last_fit_dbsize = self.dbsize

        self.model = best_model
        self.model.save_model(os.path.join(self.options['local'],
                                           f'gp_{self.name}-{self.dbsize}.json'), compress=True)

        # Also save parameters as numpy array.
        # This is more consistent across python versions (avoids pickle)
        np.save(os.path.join(self.options['local'], f'gp_{self.name}-{self.dbsize}.npy'), self.model.param_array)

        # To load it later:
        # m_load = GPRegression(np.array([[], [], []]).T, np.array([[], []]).T, initialize=False)
        # m_load.update_model(False)
        # m_load.initialize_parameter()
        # m_load[:] = np.load('model_save.npy')
        # m_load.update_model(True)

    def _similarity_check(self, ix, iy=1):
        """Check similarity between a new input point, and the existing training database
        by evaluating the kernel.

        Parameters
        ----------
        ix : int
            Location on the 2D Cartesian grid (x index)
        iy : int, optional
            Location on the 2D Cartesian grid (y index) (the default is 1)

        Returns
        -------
        bool
            True if a similar point exists.
        """

        Qnew = self.sol[:, ix, iy, None]
        Cnew = self.db.get_constant(ix, iy)
        Xnew = np.vstack([Cnew, Qnew])

        similar_point_exists = False

        for j in range(self.db.Xtrain.shape[1]):

            var = self.kern.variance
            similarity = (self.kern.K(Xnew[self.Xmask, 0, None].T, self.db.Xtrain[self.Xmask, j, None].T) / var)[0][0]

            if np.isclose(similarity, 1., rtol=0., atol=1e-8):
                similar_point_exists = True
                break

        return similar_point_exists

    def _update_database(self, ix, iy):
        """Update the training database.

        Parameters
        ----------
        ix : int
            Location on the 2D Cartesian grid (x index)
        iy : int, optional
            Location on the 2D Cartesian grid (y index) (the default is 1)
        """

        Q = self.sol[:, ix, iy]

        if Q.ndim == 1:
            Q = Q[:, None]

        self.gp_print(f'\n>>>> Database update in step {self.step + 1} ({self.name}):')

        self.db.update(Q, ix, iy)

    def gp_print(self, msg='', **kwargs):
        """
        Wrapper around print function. Prints only if verbose is True.
        """
        if self.options['verbose']:
            print(msg, **kwargs)

    @abc.abstractmethod
    def _get_test_input(self):
        """
        Assemble test input data

        Raises
        ------
        NotImplementedError
            Needs to be implemented in the derived classes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _set_solution(self, q):
        """
        Store solution in internal variable

        Parameters
        ----------
        q : numpy.ndarray
            Current solution

        Raises
        ------
        NotImplementedError
            Needs to be implemented in the derived classes.
        """

        raise NotImplementedError


class GP_stress(GaussianProcess):
    """
    Shear stress, 1D problems
    """

    ndim = 1
    name = 'shear'

    def __init__(self, db, gp):

        atol = gp['atolShear']
        noise_variance = gp['noiseShear']

        Xmask = db.num_features * [True]
        # Disable jy input
        Xmask[-1] = False

        # Outputs: tau_xz bottom, tau_xz top
        Ymask = 13 * [False]
        Ymask[5] = True
        Ymask[11] = True

        super().__init__(np.sum(Xmask), db, gp, Xmask, Ymask, atol, noise_variance)

    def _get_test_input(self):
        """
        Test data as input for GP prediction.
        Does not contain dh_dx in this case (tau_xz does not depend on dh_dx)
        """

        if self.db.mode == "gap_angle":
            Xtest_raw = np.vstack([self.db.c[0, :, 1], self.db.c[1, :, 1],
                                   self.sol[0, :, 1], self.sol[1, :, 1]])
        else:
            Xtest_raw = np.vstack([self.db.c[0, :, 1],
                                   self.sol[0, :, 1], self.sol[1, :, 1]])

        return Xtest_raw.T

    def _set_solution(self, q):
        self.sol = q


class GP_stress2D_xz(GaussianProcess):
    """
    Shear stress (xz), 2D problems
    """

    ndim = 2
    name = 'shearXZ'

    def __init__(self, db, gp):

        atol = gp['atolShear']
        noise_variance = gp['noiseShear']

        Xmask = db.num_features * [True]

        # Outputs: tau_xz bottom, tau_xz top
        Ymask = 13 * [False]
        Ymask[5] = True
        Ymask[11] = True

        super().__init__(np.sum(Xmask), db, gp, Xmask, Ymask, atol, noise_variance)

    def _get_test_input(self):
        """
        Test data as input for GP prediction.
        Does not contain dh_dx in this case (tau_xz does not depend on dh_dx)
        """

        if self.db.mode == "gap_angle":
            Xtest_raw = np.dstack([self.db.c[0], self.db.c[1],
                                   self.sol[0], self.sol[1], self.sol[2]])
        else:
            Xtest_raw = np.dstack([self.db.c[0],
                                   self.sol[0], self.sol[1], self.sol[2]])

        Xtest = np.reshape(np.transpose(Xtest_raw, (2, 0, 1)), (self.active_dim, -1)).T

        return Xtest

    def _set_solution(self, q):

        self.sol = deepcopy(q)
        self.sol[2] *= np.sign(q[2])


class GP_stress2D_yz(GaussianProcess):
    """   
    Shear stress (yz), 2D problems
    """

    ndim = 2
    name = 'shearYZ'

    def __init__(self, db, gp):

        atol = gp['atolShear']
        # kernel_init_var = gp['varShear']
        # kernel_init_scale = gp['scaleShear']
        noise_variance = gp['noiseShear']

        Xmask = db.num_features * [True]

        # Outputs: tau_yz bottom, tau_yz top
        Ymask = 13 * [False]
        Ymask[4] = True
        Ymask[10] = True

        super().__init__(np.sum(Xmask), db, gp, Xmask, Ymask, atol, noise_variance)

    def _get_test_input(self):
        """
        Test data as input for GP prediction.
        Does not contain dh_dx in this case (tau_xz does not depend on dh_dx)
        """

        if self.db.mode == "gap_angle":
            Xtest_raw = np.dstack([self.db.c[0], self.db.c[1],
                                   self.sol[0], self.sol[1], self.sol[2]])
        else:
            Xtest_raw = np.dstack([self.db.c[0],
                                   self.sol[0], self.sol[1], self.sol[2]])

        Xtest = np.reshape(np.transpose(Xtest_raw, (2, 0, 1)), (self.active_dim, -1)).T

        return Xtest

    def _set_solution(self, q):

        self.sol = deepcopy(q)
        self.sol[2] *= np.sign(q[2])


class GP_pressure(GaussianProcess):
    """
    Normal stress (pressure), 1 dimension
    """

    ndim = 1
    name = 'press'

    def __init__(self, db, gp):

        atol = gp['atolPress']
        noise_variance = gp['noisePress']

        Xmask = db.num_features * [True]
        # Disable jy
        Xmask[-1] = False

        # Output: only p
        Ymask = 13 * [False]
        Ymask[0] = True

        super().__init__(np.sum(Xmask), db, gp, Xmask, Ymask, atol, noise_variance)

    def _get_test_input(self):
        """
        Test data as input for GP prediction:
        For pressure, only density is required (not with MD!)
        """

        if self.db.mode == "gap_angle":
            Xtest_raw = np.vstack([self.db.c[0, :, 1], self.db.c[1, :, 1],
                                   self.sol[0, :, 1], self.sol[1, :, 1]])
        else:
            Xtest_raw = np.vstack([self.db.c[0, :, 1],
                                   self.sol[0, :, 1], self.sol[1, :, 1]])

        return Xtest_raw.T

    def _set_solution(self, q):
        self.sol = q  # [:, :, 1]


class GP_pressure2D(GaussianProcess):
    """
    Normal stress (pressure), 2 dimensions
    """

    ndim = 2
    name = 'press'

    def __init__(self, db, gp):

        atol = gp['atolPress']
        noise_variance = gp['noisePress']

        # All inputs
        Xmask = db.num_features * [True]

        # Output: only pressure
        Ymask = 13 * [False]
        Ymask[0] = True

        super().__init__(np.sum(Xmask), db, gp, Xmask, Ymask, atol, noise_variance)

    def _get_test_input(self):
        """
        Test data as input for GP prediction:
        For pressure, only density is required (not with MD!)
        """

        if self.db.mode == "gap_angle":
            Xtest_raw = np.dstack([self.db.c[0], self.db.c[1],
                                   self.sol[0], self.sol[1], self.sol[2]])
        else:
            Xtest_raw = np.dstack([self.db.c[0],
                                   self.sol[0], self.sol[1], self.sol[2]])

        Xtest = np.reshape(np.transpose(Xtest_raw, (2, 0, 1)), (self.active_dim, -1)).T

        return Xtest

    def _set_solution(self, q):

        self.sol = deepcopy(q)
        self.sol[2] *= np.sign(q[2])
