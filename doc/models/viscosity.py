import io
import numpy as np
import matplotlib.pyplot as plt
from importlib import resources

from GaPFlow.io import read_yaml_input
from GaPFlow.models.viscosity import shear_thinning_factor, piezoviscosity


eyring = """
properties:
    shear: 0.1
    bulk: 0.
    EOS: DH
    thinning:
        name: Eyring
"""

carreau = """
properties:
    shear: 0.1
    bulk: 0.
    EOS: DH
    thinning:
        name: Carreau
"""

barus = """
properties:
    shear: 0.1
    bulk: 0.
    EOS: DH
    piezo:
        name: Barus
"""

roelands = """
properties:
    shear: 0.1
    bulk: 0.
    EOS: DH
    piezo:
        name: Roelands
"""


if __name__ == "__main__":

    plt.style.use(resources.files("GaPFlow.resources").joinpath("gapflow.mplstyle"))

    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Viscosity models (with default parameters)')

    # Shear thinning

    shear_rate = np.logspace(0, 11, 100)

    for model in [eyring, carreau]:

        with io.StringIO(model) as file:
            inp = read_yaml_input(file)['properties']

        mu0 = inp['shear']
        name = inp['thinning']['name']

        viscosity = mu0 * shear_thinning_factor(shear_rate, mu0, inp['thinning'])

        ax[0].loglog(shear_rate, viscosity, label=name)

    ax[0].set_xlabel('Shear rate')
    ax[0].set_ylabel('Viscosity')
    ax[0].legend()

    # Piezoviscosity

    pressure = np.linspace(1e6, 1e9, 100)

    for model in [barus, roelands]:

        with io.StringIO(model) as file:
            inp = read_yaml_input(file)['properties']

        mu0 = inp['shear']
        name = inp['piezo']['name']
        viscosity = mu0 * piezoviscosity(pressure, mu0, inp['piezo'])

        ax[1].loglog(pressure, viscosity, label=name)

    ax[1].set_xlabel('Pressure')
    ax[1].set_ylabel('Viscosity')
    ax[1].legend()

    plt.show()
