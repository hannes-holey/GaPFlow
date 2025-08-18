#
# Copyright 2019, 2025 Hannes Holey
#           2019 Andrea Codrignani
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


import yaml
import numpy as np

from hans.problem import Problem
from hans.tools import abort
from hans.material import Material


class Input:

    def __init__(self, inputFile, restartFile=None):
        """
        Reads the yaml input file that defines a problem.

        Parameters
        ----------
        inputFile : str
            filename of the .yaml input file
        restartFile : str
            filename of the .nc data file (default: None)

        """
        self.inputFile = inputFile
        self.restartFile = restartFile

    def getProblem(self):
        """
        Parses the yaml input file, performs sanity checks,
        and returns an instance of the Problem class.

        Returns
        -------
        Problem
            Problem object containing all information about the current job.

        """

        with open(self.inputFile, 'r') as ymlfile:

            inp = yaml.full_load(ymlfile)

            print(f"Reading input from {self.inputFile}...")

            # Mandatory inputs
            options = self.sanitize_options(inp['options'])
            disc = self.sanitize_disc(inp['disc'])
            numerics, disc = self.sanitize_numerics(inp['numerics'], disc)
            geometry = self.sanitize_geometry(inp['geometry'], disc)
            material = self.sanitize_material(inp['material'])
            bc, disc = self.sanitize_BC(inp['BC'], disc, material)

            # Optional inputs
            surface = self.sanitize_surface(inp["surface"]) if "surface" in inp.keys() else None
            roughness = self.sanitize_roughness(inp["roughness"]) if "roughness" in inp.keys() else None
            gp = self.sanitize_gp(inp["gp"]) if 'gp' in inp.keys() else None
            md = self.sanitize_md(inp["md"]) if 'md' in inp.keys() else None

            if self.restartFile is None:
                ic = self.sanitize_IC(inp["IC"], disc) if "IC" in inp.keys() else None
            else:
                ic = {"type": "restart", "file": self.restartFile}

            print("Sanity checks completed. Start simulation!")
            print(60 * "-")

        thisProblem = Problem(options,
                              disc,
                              bc,
                              geometry,
                              numerics,
                              material,
                              surface,
                              ic,
                              roughness,
                              gp,
                              md)

        return thisProblem

    def sanitize_disc(self, disc):
        """Sanity check for discretization settings

        Parameters
        ----------
        disc : dict
            Discretization input dict

        Returns
        -------
        dict
            Sanitized discretization input dict
        """

        print("Checking discretization... ")

        mandatory_keys = ['Nx', 'Ny']
        optional_keys = ['Lx', 'dx', 'Ly', 'dy']

        # Check for Nx, Ny
        assert np.sum([key in disc.keys() for key in mandatory_keys]) == 2

        disc['Nx'] = int(disc['Nx'])
        disc['Ny'] = int(disc['Ny'])

        assert disc['Nx'] > 0
        assert disc['Ny'] > 0

        # Check for the rest
        keys_in_input = np.array([key in disc.keys() for key in optional_keys])
        assert np.sum(keys_in_input[:2]) >= 1
        assert np.sum(keys_in_input[2:]) >= 1

        if keys_in_input[0]:
            disc['Lx'] = float(disc['Lx'])
            disc['dx'] = float(disc["Lx"]) / disc['Nx']
        else:
            disc['Lx'] = float(disc["dx"]) * disc['Nx']

        if keys_in_input[2]:
            disc['Ly'] = float(disc['Ly'])
            disc['dy'] = float(disc["Ly"]) / disc['Ny']
        else:
            disc['Ly'] = float(disc["dy"]) * disc['Ny']

        return disc

    def sanitize_options(self, options):
        """Sanitize output settings


        Parameters
        ----------
        options : dict
            Output settings

        Returns
        -------
        dict
            Sanitized output settings
        """
        print("Checking I/O options... ")

        keys = ['writeInterval', 'writeRestart']
        types = [int, int]
        defaults = [1000, 0]

        for k, t, d in zip(keys, types, defaults):
            if k in options.keys():
                options[k] = abs(t(options[k]))
            else:
                options[k] = d

        return options

    def sanitize_geometry(self, geometry, disc):
        """Sanitize geometry

        Parameters
        ----------
        geometry : dict
            Geometry settings
        disc : dict
            Discretization settings

        Returns
        -------
        dict
            Sanitized geometry settings
        """

        print("Checking geometry... ")

        if geometry["type"] in ["journal", "journal_x", "journal_y"]:
            geometry["CR"] = float(geometry["CR"])
            geometry["eps"] = float(geometry["eps"])
        elif geometry["type"] == "parabolic":
            geometry["hmin"] = float(geometry['hmin'])
            geometry["hmax"] = float(geometry['hmax'])
        elif geometry["type"] == "asperity":
            geometry["hmin"] = float(geometry['hmin'])
            geometry["hmax"] = float(geometry['hmax'])
            geometry["nperside"] = int(geometry['nperside'])
        elif geometry["type"] == "twin_parabolic":
            geometry["hmin"] = float(geometry['hmin'])
            geometry["hmax"] = float(geometry['hmax'])
        elif geometry["type"] in ["inclined", "inclined_x", "inclined_y"]:
            geometry["h1"] = float(geometry['h1'])
            geometry["h2"] = float(geometry['h2'])
        elif geometry["type"] in ["cdc"]:
            geometry["h0"] = float(geometry['h0'])
            # geometry["h1"] = float(geometry['h1'])
            geometry["b"] = float(geometry['b'])
        elif geometry["type"] == "inclined_pocket":
            geometry["h1"] = float(geometry['h1'])
            geometry["h2"] = float(geometry['h2'])
            geometry["hp"] = float(geometry['hp'])
            geometry["c"] = float(geometry['c'])
            geometry["l"] = float(geometry['l'])
            geometry["w"] = float(geometry['w'])

            try:
                geometry["t"] = float(geometry["t"])
            except KeyError:
                geometry["t"] = 0.

            try:
                assert 2 * geometry["t"] + geometry["l"] + geometry["c"] <= disc["Lx"]
                assert geometry["w"] <= disc["Ly"]
            except AssertionError:
                print("Size of the pocket is larger than the domain. Abort.")
                abort()

        elif geometry["type"] in ["half_sine", "half_sine_squared"]:
            geometry["h0"] = float(geometry['h0'])
            geometry["amp"] = float(geometry['amp'])
            geometry["num"] = float(geometry['num'])

        else:
            print(f"'{geometry['type']}' geometry not implemented")
            abort()

        return geometry

    def sanitize_numerics(self, numerics, disc):
        """Sanitize numerics settings

        Parameters
        ----------
        numerics : dict
            Numerics input settings
        disc : dict
            Discretization settings

        Returns
        -------
        dict
            Sanitized numerics settings
        dict
            Modified discretization settings
        """

        print("Checking numerics options... ")

        try:
            numerics["integrator"] = numerics["integrator"]
            assert numerics["integrator"] in ["MC", "MC_bf", "MC_fb", "MC_alt", "LW", "RK3"]
        except KeyError:
            print("***Integrator not specified. Use default (MacCormack).")
            numerics["integrator"] = "MC"
        except AssertionError:
            print(f'***Unknown integrator \'{numerics["integrator"]}\'. Abort.')
            abort()

        if numerics["integrator"].startswith("MC"):
            try:
                numerics["fluxLim"] = float(numerics["fluxLim"])
            except KeyError:
                pass

        try:
            numerics["stokes"] = int(numerics["stokes"])
        except KeyError:
            print("***Boolean parameter 'stokes' not given. Use default (True).")
            numerics["stokes"] = 1

        try:
            numerics["adaptive"] = int(numerics["adaptive"])
        except KeyError:
            print("***Boolean parameter 'adaptive' not given. Use default (False).")
            numerics["adaptive"] = 0

        if numerics["adaptive"] == 1:
            try:
                numerics["C"] = float(numerics["C"])
            except KeyError:
                print("***CFL number not given. Use default (0.5).")
                numerics["C"] = 0.5

        try:
            numerics["dt"] = float(numerics["dt"])
        except KeyError:
            print("***Timestep not given. Use default (1e-10).")
            numerics["dt"] = 1e-10

        stopping_criteria = 0

        try:
            numerics["tol"] = float(numerics["tol"])
            stopping_criteria += 1
        except KeyError:
            pass

        try:
            numerics["maxT"] = float(numerics["maxT"])
            stopping_criteria += 1
        except KeyError:
            pass

        try:
            numerics["maxIt"] = int(numerics["maxIt"])
            stopping_criteria += 1
        except KeyError:
            pass

        if stopping_criteria == 0:
            print("***No stopping criterion given. Abort.")
            abort()

        if numerics["integrator"] == "RK3":
            disc["nghost"] = 2
        else:
            disc["nghost"] = 1

        return numerics, disc

    def sanitize_material(self, material):
        """Sanitize material settings

        Parameters
        ----------
        material : dict
            Material (fluid) properties

        Returns
        -------
        dict
            Sanitized material properties
        """

        print("Checking material options... ")

        if material["EOS"] == "DH":
            keys = ['rho0', 'P0', 'C1', 'C2']
            types = [float, float, float, float]
            material = check_input(material, keys, types)

        elif material["EOS"] == "PL":
            keys = ['rho0', 'P0', 'alpha']
            types = [float, float, float]
            material = check_input(material, keys, types)

        elif material["EOS"] == "vdW":
            keys = ['M', 'T0', 'a', 'b']
            types = [float, float, float, float]
            material = check_input(material, keys, types)

        elif material["EOS"] == "Tait":
            keys = ['rho0', 'P0', 'K', 'n']
            types = [float, float, float, float]
            material = check_input(material, keys, types)

        elif material["EOS"] == "cubic":
            keys = ['a', 'b', 'c', 'd']
            types = [float, float, float, float]
            material = check_input(material, keys, types)

        elif material["EOS"].startswith("Bayada"):
            keys = ['cl', 'cv', 'rhol', 'rhov', 'shear', 'shearv', 'rhov']
            types = [float, float, float, float, float, float, float]
            material = check_input(material, keys, types)

        elif material['EOS'] == 'BWR':
            material['T'] = float(material['T'])
            material['rho0'] = float(material['rho0'])
            # Parameters: Johnson et al., Mol. Phys. 78 (1993)
            params = """
                    0.8623085097507421
                    2.976218765822098
                    -8.402230115796038
                    0.1054136629203555
                    -0.8564583828174598
                    1.582759470107601
                    0.7639421948305453
                    1.753173414312048
                    2.798291772190376e+03
                    -4.8394220260857657e-02
                    0.9963265197721935
                    -3.698000291272493e+01
                    2.084012299434647e+01
                    8.305402124717285e+01
                    -9.574799715203068e+02
                    -1.477746229234994e+02
                    6.398607852471505e+01
                    1.603993673294834e+01
                    6.805916615864377e+01
                    -2.791293578795945e+03
                    -6.245128304568454
                    -8.116836104958410e+03
                    1.488735559561229e+01
                    -1.059346754655084e+04
                    -1.131607632802822e+02
                    -8.867771540418822e+03
                    -3.986982844450543e+01
                    -4.689270299917261e+03
                    2.593535277438717e+02
                    -2.694523589434903e+03
                    -7.218487631550215e+02
                    1.721802063863269e+02
                    """

            x = [float(val) for val in params.split()]
            material['params'] = x

        elif material["EOS"] == "MD":
            # Density for initial conditions, everything else in "md" input section
            keys = ['rho0', ]
            types = [float, ]
            material = check_input(material, keys, types)

        if material["EOS"] != "MD":
            material["shear"] = float(material["shear"])
            material["bulk"] = float(material["bulk"])

        if "Pcav" in material.keys():
            material["Pcav"] = float(material["Pcav"])

        if "piezo" in material.keys():
            if material["piezo"] == "Barus":
                material["aB"] = float(material["aB"])
            elif material["piezo"] == "Vogel":
                material["rho0"] = float(material['rho0'])
                material["g"] = float(material["g"])
                material["mu_inf"] = float(material["mu_inf"])
                material["phi_inf"] = float(material["phi_inf"])
                material["BF"] = float(material["BF"])

        if "thinning" in material.keys():
            if material["thinning"] == "Eyring":
                material["tau0"] = float(material["tau0"])
            elif material["thinning"] == "Carreau":
                material["relax"] = float(material["relax"])
                material["a"] = float(material["a"])
                material["N"] = float(material["N"])
            elif material["thinning"] == "PL":
                material["shear"] = float(material["shear"])
                material["n"] = float(material["n"])

        if "PLindex" in material.keys():
            material["PLindex"] = float(material["PLindex"])
            if "PLmethod" in material.keys():
                material["PLmethod"] = material["PLmethod"]
            else:
                material["PLmethod"] = "exact"

        return material

    def sanitize_BC(self, bc, disc, material):
        """Sanity check for boundary condition input.

        Parameters
        ----------
        bc : dict
            Boundary condition parameters read from yaml input file.
        disc : dict
            Discretization parameters.
        material : dict
            Material parameters.

        Returns
        -------
        dict
            Boundary condition parameters.
        dict
            Modified discretization parameters.

        """
        print("Checking boundary conditions... ")

        bc["x0"] = np.array(list(bc["x0"]))
        bc["x1"] = np.array(list(bc["x1"]))
        bc["y0"] = np.array(list(bc["y0"]))
        bc["y1"] = np.array(list(bc["y1"]))

        assert len(bc["x0"]) == 3
        assert len(bc["x1"]) == 3
        assert len(bc["y0"]) == 3
        assert len(bc["y1"]) == 3

        disc['pX'] = int("P" in bc["x0"] and "P" in bc["x1"])
        disc['pY'] = int("P" in bc["y0"] and "P" in bc["y1"])

        if "D" in bc["x0"]:
            if "px0" in bc.keys():
                bc["rhox0"] = Material(material).eos_density(float(bc["px0"]))
            else:
                bc["rhox0"] = float(bc["rhox0"]) if 'rhox0' in bc.keys() else material["rho0"]

        if "D" in bc["x1"]:
            if "px1" in bc.keys():
                bc["rhox1"] = Material(material).eos_density(float(bc["px1"]))
            else:
                bc["rhox1"] = float(bc["rhox1"]) if 'rhox1' in bc.keys() else material['rho0']

        if "D" in bc["y0"]:
            if "py0" in bc.keys():
                bc["rhoy0"] = Material(material).eos_density(float(bc["py0"]))
            else:
                bc["rhoy0"] = float(bc["rhoy0"]) if 'rhoy0' in bc.keys() else material["rho0"]

        if "D" in bc["y1"]:
            if "py1" in bc.keys():
                bc["rhoy1"] = Material(material).eos_density(float(bc["py1"]))
            else:
                bc["rhoy1"] = float(bc["rhoy1"]) if 'rhoy1' in bc.keys() else material['rho0']

        assert np.all((bc["x0"] == "P") == (bc["x1"] == "P")), "Inconsistent boundary conditions (x)"
        assert np.all((bc["y0"] == "P") == (bc["y1"] == "P")), "Inconsistent boundary conditions (y)"

        return bc, disc

    def sanitize_surface(self, surface):
        """Sanitize surface parameters

        Parameters
        ----------
        surface : dict
            Input surface parameters

        Returns
        -------
        dict
            Sanitized surface parameters
        """
        print("Checking surface parameters... ")

        if "lslip" in surface.keys():
            surface["lslip"] = float(surface["lslip"])
        else:
            surface["lslip"] = 0.

        if surface["type"] in ["stripes", "stripes_x", "stripes_y"]:
            try:
                surface["num"] = int(surface["num"])
            except KeyError:
                surface["num"] = 1
            try:
                surface["sign"] = int(surface["sign"])
            except KeyError:
                surface["sign"] = -1

        return surface

    def sanitize_roughness(self, roughness):
        """Sanitize roughness settings

        Parameters
        ----------
        roughness : dict
            Roughness settings

        Returns
        -------
        dict
            Sanitized roughness settings
        """

        print("Checking roughness parameters... ")

        if "file" not in roughness.keys():
            try:
                roughness["seed"] = int(roughness["seed"])
            except KeyError:
                roughness["seed"] = None
            try:
                roughness["Hurst"] = float(roughness["Hurst"])
            except KeyError:
                "Hurst exponent not given. Use default (0.8)."
                roughness["Hurst"] = 0.8
            try:
                roughness["rolloff"] = float(roughness["rolloff"])
            except KeyError:
                "Rollof not given. Use default (1.0)"
                roughness["rolloff"] = 1.0
            try:
                roughness["rmsHeight"] = float(roughness["rmsHeight"])
            except KeyError:
                roughness["rmsHeight"] = None
            try:
                roughness["rmsSlope"] = float(roughness["rmsSlope"])
            except KeyError:
                roughness["rmsSlope"] = None
            try:
                roughness["shortCutoff"] = float(roughness["shortCutoff"])
            except KeyError:
                roughness["shortCutoff"] = None
            try:
                roughness["longCutoff"] = float(roughness["longCutoff"])
            except KeyError:
                roughness["longCutoff"] = None

            if roughness["rmsHeight"] is None and roughness["rmsSlope"] is None:
                print('Neither rms height nor rms slope is defined! Abort.')
                abort()

        return roughness

    def sanitize_IC(self, ic, disc):
        """Sanitize initial conditions

        Parameters
        ----------
        ic : dict
            Initial conditions settings
        disc : dict
            Discretization settings

        Returns
        -------
        dict
            Sanitized initial conditions setting
        """
        print("Checking initial conditions... ")

        if ic["type"] != "restart":
            if ic["type"] == "perturbation":
                ic["factor"] = float(ic["factor"])
            elif ic["type"] in ["longitudinal_wave", "shear_wave"]:
                ic["amp"] = float(ic["amp"])
                if "nwave" in ic.keys():
                    ic["nwave"] = int(ic["nwave"])
                else:
                    ic["nwave"] = 1
            elif ic["type"] == "random":
                try:
                    ic["stdDens"] = float(ic["stdDens"])
                except KeyError:
                    ic["stdDens"] = 0.0
                try:
                    ic["stdFlux"] = float(ic["stdFlux"])
                except KeyError:
                    ic["stdFlux"] = 0.0
            elif ic["type"] == "gauss1D":
                try:
                    ic["mean"] = float(ic["mean"])
                except KeyError:
                    ic["mean"] = float(disc["Lx"]) / 2
                try:
                    ic["stdev"] = float(ic["stdev"])
                except KeyError:
                    ic["stdev"] = float(disc["Lx"]) / 40
                try:
                    ic["cutoff"] = float(ic["cutoff"])
                except KeyError:
                    ic["cutoff"] = 12 * ic['stdev']

        return ic

    def sanitize_gp(self, gp):
        print("Checking GP parameters... ")

        # Tolerances
        mandatory_keys = ['atolShear', 'atolPress']
        mandatory_types = len(mandatory_keys) * [float]
        gp = check_input(gp, mandatory_keys, mandatory_types)

        # Rtol, kernel and noise
        optional_gp_keys = ['rtol',                                                         # Relative tolerance
                            # 'scaleShear', 'scalePress', 'varShear', 'varPress',             # Kernel
                            'noiseShear', 'noisePress', 'noiseFixed', 'heteroscedastic',    # Noise
                            'optRestarts', 'maxSteps', 'maxResets', 'wait', 'verbose']      # Options

        # optional_gp_types = [float] + 2 * [list] + 4 * [float] + 7 * [int]
        optional_gp_types = 3 * [float] + 7 * [int]
        optional_gp_defaults = [0.05,
                                # [], [], -1., -1.,
                                0., 0., 0, 0, 10, 5, 5, 100, 1]
        gp = check_input(gp, optional_gp_keys, optional_gp_types, optional_gp_defaults)

        # DB stuff
        optional_db_keys = ['remote', 'local', 'storage', 'Ninit', 'sampling']
        optional_db_types = [int, str, str, int, str]
        optional_db_defaults = [0, '/tmp/dtool/', 's3://test-bucket', 5, 'lhc']

        gp = check_input(gp, optional_db_keys, optional_db_types, optional_db_defaults)

        return gp

    def sanitize_md(self, md):
        print("Checking MD parameters... ")

        keys = ['ncpu', 'infile', 'wallfile', 'cutoff', 'temp', 'vWall', 'tsample']
        types = [int, str, str, float, float, float, int]

        # md = check_input(md, keys, types)

        return md


def check_input(params, keys, types, defaults=None):

    if defaults is None:
        assert np.all([key in params.keys() for key in keys])
        for k, t in zip(keys, types):
            if k in params.keys():
                params[k] = t(params[k])
    else:
        for k, t, d in zip(keys, types, defaults):
            if k in params.keys():
                params[k] = t(params[k])
            else:
                params[k] = d

    return params
