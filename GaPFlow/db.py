import os
import jax.numpy as jnp
import jax.random as jr

from GaPFlow.models.pressure import eos_pressure
from GaPFlow.models.viscous import stress_bottom, stress_top
from GaPFlow.dtool import get_readme_list_local, init_dataset, write_readme


class Database:

    def __init__(self, minimum_size,
                 Xtrain=jnp.empty((0, 6)),
                 Ytrain=jnp.empty((0, 13)),
                 Yerr=jnp.empty((0, 3)),
                 outdir=None):

        self.outdir = outdir
        self.minimum_size = minimum_size

        self._Xtrain = Xtrain
        self._Ytrain = Ytrain

        if self.size == 0:
            self.X_scale = jnp.ones((6,))
            self.Y_scale = jnp.ones((13,))
        else:
            self.X_scale = self.normalizer(self._Xtrain)  # shape=(6, )
            self.Y_scale = self.normalizer(self._Ytrain)  # shape=(13, )

    # TODO
    @classmethod
    def from_numpy(cls, directory):
        raise NotImplementedError

    # TODO
    @classmethod
    def from_dtool(cls, directory, minimum_size, outdir):
        readme_list = get_readme_list_local(directory)
        cls.dtool_basepath = directory

        if len(readme_list) > 0:

            Xtrain = []
            Ytrain = []
            Yerr = []

            for rm in readme_list:
                X = jnp.array(rm['X'])
                Y = jnp.array(rm['Y'])
                Xtrain.append(X)
                Ytrain.append(Y)

                if 'Yerr' in rm.keys():
                    Yerr.append(rm['Yerr'])
                else:
                    Yerr.append([0., 0., 0.])

            Xtrain = jnp.array(Xtrain)
            Ytrain = jnp.array(Ytrain)
            Yerr = jnp.array(Yerr)

        else:
            print("No matching dtool datasets found. Start with empty database.")
            Xtrain = jnp.empty((0, 6))
            Ytrain = jnp.empty((0, 13))
            Yerr = jnp.empty((0, 3))

        return cls(minimum_size, Xtrain, Ytrain, Yerr, outdir)

    def fill_missing(self, Xtest, prop=None, geo=None, noise=(0., 0.)):

        Xnew = get_new_training_input(Xtest.T,
                                      self.minimum_size - self.size)

        self.add_data(Xnew, prop=prop, geo=geo, noise=noise)

    def add_data(self, Xnew, prop=None, geo=None, noise=(0., 0.), dtool=True):
        size_before = self.size

        if prop is not None:
            # Ynew from "mock" MD
            Ynew = get_new_training_output_mock(Xnew, prop, geo,
                                                noise_stddev=noise)
            Yerr = jnp.zeros_like(Ynew)
        else:
            # Ynew, Yerr from actual MD
            pass

        self._Xtrain = jnp.vstack([self._Xtrain, Xnew.T])
        self._Ytrain = jnp.vstack([self._Ytrain, Ynew.T])

        self.X_scale = self.normalizer(self._Xtrain)
        self.Y_scale = self.normalizer(self._Ytrain)

        # write dtool datasets
        # with MD, loop needs to enclose LAMMPS run

        if dtool:
            for X, Y, Ye in zip(Xnew.T, Ynew.T, Yerr.T):
                size_before += 1
                proto_ds, proto_ds_path = init_dataset(self.dtool_basepath, size_before)

                write_readme(proto_ds_path, X, Y, Ye)
                proto_ds.freeze()

        self.write()

    def normalizer(self, x):
        # TODO: different normalizers
        return jnp.maximum(jnp.max(jnp.abs(x), axis=0), 1e-12)

    @property
    def Xtrain(self):
        return self._Xtrain / self.X_scale

    @property
    def Ytrain(self):
        return self._Ytrain / self.Y_scale

    @property
    def size(self):
        return self._Xtrain.shape[0]

    def write(self):
        if self.outdir is not None:
            jnp.save(os.path.join(self.outdir, 'Xtrain.npy'), self._Xtrain)
            jnp.save(os.path.join(self.outdir, 'Ytrain.npy'), self._Ytrain)


# Mock MD data

def get_new_training_input(Xtest, Nsample, width=1e-2):

    if Nsample > 0:
        # Bounds for quasi random sampling of initial database
        jabs = jnp.hypot(jnp.mean(Xtest[4, :]), jnp.mean(Xtest[5, :]))
        rho = jnp.mean(Xtest[3, :])

        l_bounds = jnp.array([(1. - width) * rho, 0.5 * jabs, 0.])
        u_bounds = jnp.array([(1. + width) * rho, 1.5 * jabs, 0.5 * jabs])

        dim = len(l_bounds)

        # Constant properties are randomly sampled collectively (e.g. from the available profiles)
        key = jr.key(123)
        key, subkey = jr.split(key)
        choice = jr.choice(key, Xtest.shape[1], shape=(Nsample,), replace=False).tolist()

        # random samples (other strategies later...)
        scaled_samples = jr.uniform(key, shape=(Nsample, dim),
                                    minval=l_bounds[None, :],
                                    maxval=u_bounds[None, :]).T

        Xnew = jnp.vstack([
            Xtest[:3, choice],  # gap height
            scaled_samples[0],  # density
            scaled_samples[1],  # flux_x
            scaled_samples[2],  # flux_y
        ])

    else:
        Xnew = jnp.empty((6, 0))

    return Xnew


def get_new_training_output_mock(X, prop, geo, noise_stddev=(0., 0.)):

    key = jr.key(123)
    key, subkey = jr.split(key)
    noise_p = jr.normal(key, shape=X.shape[1]) * noise_stddev[0]
    noise_s0 = jr.normal(key, shape=X.shape[1]) * noise_stddev[1]
    noise_s1 = jr.normal(key, shape=X.shape[1]) * noise_stddev[1]

    # For MD data: call update method from database (or external)

    # Shear stress
    U = geo['U']
    V = geo['V']
    eta = prop['shear']
    zeta = prop['bulk']

    tau_bot = stress_bottom(X[3:],  # q
                            X[:3],  # h, dhdx, dhdy
                            U, V, eta, zeta, 0.) + noise_s0

    tau_top = stress_top(X[3:],  # q
                         X[:3],  # h, dhdx, dhdy
                         U, V, eta, zeta, 0.) + noise_s1

    # Pressure
    press = eos_pressure(X[3], prop)[None, :] + noise_p

    return jnp.vstack([press,
                       tau_bot,
                       tau_top])
