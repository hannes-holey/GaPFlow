#!/bin/bash

set -e

BUILDDIR=lammps

mkdir -p ${BUILDDIR}/build

cmake ${BUILDDIR}/cmake -B${BUILDDIR}/build \
		-DBUILD_SHARED_LIBS=yes \
		-DLAMMPS_MACHINE=mpi \
		-DBUILD_LIB=yes \
		-DBUILD_LAMMPS_PYTHON=yes \
		-DCMAKE_BUILD_TYPE=Release \
		-DPKG_MOLECULE=yes \
		-DPKG_MANYBODY=yes \
		-DPKG_EXTRA-FIX=yes

cmake --build ${BUILDDIR}/build -j 8

make -C ${BUILDDIR}/build install

python3 -m pip uninstall -y lammps
python3 -m pip install mpi4py

make -C ${BUILDDIR}/build install-python

python3 .check_lammps.py
