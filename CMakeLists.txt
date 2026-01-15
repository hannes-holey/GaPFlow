cmake_minimum_required(VERSION 3.21)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 14.0 CACHE STRING "" FORCE)
endif()

project(GaPFlow LANGUAGES NONE)

include(ExternalProject)

ExternalProject_Add(lammps_ext
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/lammps
    SOURCE_SUBDIR cmake
    BINARY_DIR ${CMAKE_BINARY_DIR}/lammps-build
    INSTALL_DIR ${CMAKE_BINARY_DIR}/lammps-install

    CMAKE_ARGS
        -B<BINARY_DIR>
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=yes
        -DCMAKE_BUILD_TYPE=Release
        -DLAMMPS_MACHINE=mpi
        -DPKG_MOLECULE=yes
        -DPKG_MANYBODY=yes
        -DPKG_EXTRA-FIX=yes

    BUILD_COMMAND ${CMAKE_COMMAND} --build .
)

# Determine platform-specific LAMMPS shared library names
if(APPLE)
    set(LAMMPS_SHARED_LIBS
        ${CMAKE_BINARY_DIR}/lammps-build/liblammps_mpi.0.dylib
        ${CMAKE_BINARY_DIR}/lammps-build/liblammps_mpi.dylib
    )
elseif(WIN32)
    set(LAMMPS_SHARED_LIBS
        ${CMAKE_BINARY_DIR}/lammps-build/Release/lammps_mpi.lib
        ${CMAKE_BINARY_DIR}/lammps-build/Release/liblammps_mpi.dll
    )
else()
    set(LAMMPS_SHARED_LIBS
        ${CMAKE_BINARY_DIR}/lammps-build/liblammps_mpi.so.0
        ${CMAKE_BINARY_DIR}/lammps-build/liblammps_mpi.so
    )
endif()

# Install LAMMPS shared libraries into vendored location
install(
    FILES ${LAMMPS_SHARED_LIBS}
    DESTINATION GaPFlow/_vendor/lammps
)