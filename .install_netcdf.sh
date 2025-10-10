#!/bin/bash

BUILDDIR=~/opt/
PREFIX=~/.local

# Compile on so may cores
NCORES=8

export HDF5_VERSION=1.14.6
export PNETCDF_VERSION=1.14.0
export NETCDF4_VERSION=4.9.3
export NETCDF4_PYTHON_VERSION=1.7.2
export MPI4PY_VERSION=4.1.0

# Install parallel version of the NetCDF library from the sources.
# This is necessary because parallel compiles (if existing) are broken on most distributions.
rm -rf ${BUILDDIR}/pnetcdf-${PNETCDF_VERSION}
mkdir -p ${BUILDDIR}/pnetcdf-${PNETCDF_VERSION}
wget -qO - https://parallel-netcdf.github.io/Release/pnetcdf-${PNETCDF_VERSION}.tar.gz | tar -xzC ${BUILDDIR}
cd ${BUILDDIR}/pnetcdf-${PNETCDF_VERSION}
./configure --disable-fortran --disable-cxx --enable-shared --prefix=${PREFIX}
make -j $NCORES 
make install

# Install HDF5
rm -rf ${BUILDDIR}/hdf5-hdf5-${HDF5_VERSION}
mkdir -p ${BUILDDIR}/hdf5-hdf5-${HDF5_VERSION}
wget -qO - https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-${HDF5_VERSION}.tar.gz | tar -xzC ${BUILDDIR}
cd ${BUILDDIR}/hdf5-hdf5-${HDF5_VERSION}
./configure --enable-parallel --prefix=${PREFIX}
make -j $NCORES
make install

# We need to compile NetCDF ourselves because there is no package that has
# parallel PnetCDF and HDF5 enabled.
mkdir -p ${BUILDDIR}/netcdf-c-build
wget -qO - https://github.com/Unidata/netcdf-c/archive/refs/tags/v${NETCDF4_VERSION}.tar.gz | tar -xzC ${BUILDDIR}
cd ${BUILDDIR}/netcdf-c-build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpicxx -DUSE_PARALLEL=ON -DNETCDF_ENABLE_PARALLEL4=ON \
-DNETCDF_ENABLE_PNETCDF=ON -DNETCDF_ENABLE_TESTS=OFF ${BUILDDIR}/netcdf-c-${NETCDF4_VERSION}
make -j $NCORES
make install

# Alternative to the above: ubuntu system packages (TODO: check if that works as well)
# # sudo apt install -y openmpi-bin libopenmpi-dev libhdf5-mpi-dev libpnetcdf-dev libnetcdf-pnetcdf-dev

# Install mpi4py
python3 -m  pip install --force-reinstall --no-binary mpi4py mpi4py==${MPI4PY_VERSION}

# Install netcdf4-python and make sure that it is compiled (no-binary),
# otherwise it will not have parallel support.
NETCDF4_DIR=${PREFIX} HDF5_DIR=${PREFIX} CC=mpicc python3 -m pip install --force-reinstall \
--no-build-isolation --no-binary netCDF4 netCDF4==${NETCDF4_PYTHON_VERSION}

