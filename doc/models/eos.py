from importlib import resources
import numpy as np
import matplotlib.pyplot as plt

from GaPFlow.models.pressure import eos_pressure

plt.style.use(resources.files("GaPFlow.resources").joinpath("gapflow.mplstyle"))


def plot_bwr(ax):
    props = {'EOS': 'BWR'}
    dens = np.linspace(0.2, 0.8, 100)

    for temp in [1.5, 2.0, 2.5]:
        props['T'] = temp
        p = eos_pressure(dens, props)

        ax.plot(dens, p, label=rf"$T=${temp}")

    ax.legend()
    ax.set_xlabel('Density')
    ax.set_ylabel('Pressure')
    ax.set_title('BWR (LJ)')


def plot_dh(ax):

    props = {'EOS': 'DH'}
    dens = np.linspace(870., 1050., 100)
    p = eos_pressure(dens, props)

    ax.plot(dens, p, label='Dowson-Higginson')

    ax.set_xlabel('Density')
    ax.set_ylabel('Pressure')
    ax.set_title('DH')


def plot_mt(ax):

    props = {'EOS': 'MT'}
    dens = np.linspace(700., 1200., 100)
    p = eos_pressure(dens, props)

    ax.plot(dens, p, label='Murnaghan-Tait')

    ax.set_xlabel('Density')
    ax.set_ylabel('Pressure')
    ax.set_title('MT')


def plot_pl(ax):
    props = {'EOS': 'PL'}
    dens = np.linspace(0., 2., 100)

    for alpha in [0., 0.5, 1., 1.5]:
        props['alpha'] = alpha
        p = eos_pressure(dens, props)
        ax.plot(dens, p, label=rf'$\alpha=${alpha:.1f}')

    ax.set_xlabel('Density')
    ax.set_ylabel('Pressure')
    ax.set_title('Power law')
    ax.legend()


def plot_vdW(ax):
    props = {'EOS': 'vdW'}

    dens = np.linspace(0., 1000., 100)
    for temp in [120., 150., 180., 210.]:
        props['T'] = temp
        p = eos_pressure(dens, props)
        ax.plot(dens, p, label=rf"$T=${temp:.0f} K")

    ax.set_xlabel('Density')
    ax.set_ylabel('Pressure')
    ax.set_title('van der Waals')
    ax.legend()


def plot_cubic(ax):
    props = {'EOS': 'cubic'}
    dens = np.linspace(0.2, 0.6, 100)

    p = eos_pressure(dens, props)

    ax.plot(dens, p, '--', color='0.0', label='cubic (T=2)')
    ax.legend()


if __name__ == "__main__":

    sx, sy = plt.rcParams['figure.figsize']
    funcs = [plot_bwr, plot_dh, plot_mt, plot_pl, plot_vdW]

    n = len(funcs)
    fig, axes = plt.subplots(n, figsize=(sx, n * sy))

    for ax, func in zip(axes, funcs):
        func(ax)

    plot_cubic(axes[0])

    plt.show()
