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

import ctypes
import os
import platform

from importlib.metadata import distribution, PackageNotFoundError

# ---------------------------------------------------------------------
# Locate vendored LAMMPS package and shared library
# ---------------------------------------------------------------------


def _installed_package_root(pkg_name: str) -> str | None:
    try:
        dist = distribution(pkg_name)
    except PackageNotFoundError:
        return None

    # Find where the package files live
    for file in dist.files or []:
        if file.parts and file.parts[0] == pkg_name:
            return os.path.join(dist.locate_file(file).parents[len(file.parts) - 1])
    return None


# ---------------------------------------------------------
# Decide where to load from
# ---------------------------------------------------------

_pkg_name = "GaPFlow"

_installed_root = _installed_package_root(_pkg_name)

if _installed_root is None:
    raise ImportError(
        "mypkg must be installed to use LAMMPS.\n"
        "Run: pip install ."
    )

_vendor_root = os.path.join(_installed_root, _pkg_name, "_vendor")

if not os.path.isdir(_vendor_root):
    raise ImportError("Installed mypkg does not contain vendored LAMMPS")

_lammps_pkg_dir = os.path.join(_vendor_root, 'lammps')


# ---------------------------------------------------------------------
# Load shared library (platform-specific)
# ---------------------------------------------------------------------

system = platform.system()

if system == "Linux":
    lib_names = [
        "liblammps.so",
        "liblammps_mpi.so",
    ]
    loader = lambda p: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)

elif system == "Darwin":
    lib_names = [
        "liblammps.dylib",
        "liblammps_mpi.dylib",
    ]
    loader = lambda p: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)

elif system == "Windows":
    lib_names = [
        "liblammps.dll",
        "liblammps_mpi.dll",
    ]

    # Required on Python >= 3.8
    os.add_dll_directory(_lammps_pkg_dir)
    loader = lambda p: ctypes.WinDLL(p)

else:
    raise OSError(f"Unsupported platform: {system}")

_lib_loaded = False
for name in lib_names:
    lib_path = os.path.join(_lammps_pkg_dir, name)
    if os.path.exists(lib_path):
        loader(lib_path)
        _lib_loaded = True
        break


if not _lib_loaded:
    raise OSError(
        "LAMMPS shared library not found in vendored directory:\n"
        f"  {_vendor_root}"
    )

# ---------------------------------------------------------------------
# Import Python bindings AFTER library is loaded
# ---------------------------------------------------------------------

from GaPFlow._vendor import lammps  # noqa: E402

__all__ = ["lammps"]
