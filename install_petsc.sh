#!/bin/bash
#
# Install PETSc and petsc4py for GaPFlow 2D FEM solver
#
# Prerequisites:
#   - MPI installed (openmpi-bin, libopenmpi-dev on Debian/Ubuntu)
#   - C compiler (gcc)
#   - Python 3.10+ with pip
#   - make, wget
#
# Usage:
#   bash install_petsc.sh
#
# This script will:
#   1. Download PETSc source
#   2. Configure with automatic dependency download (BLAS, LAPACK, MUMPS)
#   3. Build PETSc
#   4. Install petsc4py
#   5. Verify installation
#

set -e

PETSC_VERSION=3.22.2
PETSC_INSTALL_DIR="${PETSC_INSTALL_DIR:-$HOME/.local/petsc-$PETSC_VERSION}"
PETSC_ARCH=arch-linux-c-opt
NPROC="${NPROC:-$(nproc 2>/dev/null || echo 4)}"

echo "=============================================="
echo "PETSc Installation Script for GaPFlow"
echo "=============================================="
echo "PETSc version: $PETSC_VERSION"
echo "Install directory: $PETSC_INSTALL_DIR"
echo "Build parallelism: $NPROC"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v mpicc &> /dev/null; then
    echo "ERROR: mpicc not found. Please install MPI first:"
    echo "  Ubuntu/Debian: sudo apt-get install openmpi-bin libopenmpi-dev"
    echo "  Fedora/RHEL:   sudo dnf install openmpi openmpi-devel"
    echo "  macOS:         brew install open-mpi"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found."
    exit 1
fi

if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
    echo "ERROR: wget or curl required for downloading PETSc."
    exit 1
fi

echo "Prerequisites OK."
echo ""

# Download PETSc
PETSC_TARBALL="petsc-$PETSC_VERSION.tar.gz"
PETSC_URL="https://web.cels.anl.gov/projects/petsc/download/release-snapshots/$PETSC_TARBALL"

if [ -d "$PETSC_INSTALL_DIR" ]; then
    echo "PETSc directory already exists: $PETSC_INSTALL_DIR"
    read -p "Remove and reinstall? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PETSC_INSTALL_DIR"
    else
        echo "Aborting."
        exit 1
    fi
fi

echo "Downloading PETSc $PETSC_VERSION..."
TMPDIR=$(mktemp -d)
cd "$TMPDIR"

if command -v wget &> /dev/null; then
    wget -q --show-progress "$PETSC_URL"
else
    curl -L -o "$PETSC_TARBALL" "$PETSC_URL"
fi

echo "Extracting..."
tar xf "$PETSC_TARBALL"
mv "petsc-$PETSC_VERSION" "$PETSC_INSTALL_DIR"
cd "$PETSC_INSTALL_DIR"
rm -rf "$TMPDIR"

# Configure PETSc
echo ""
echo "Configuring PETSc (this may take a few minutes)..."
echo "Downloading and building dependencies as needed..."

./configure \
    PETSC_ARCH="$PETSC_ARCH" \
    --with-cc=mpicc \
    --with-cxx=mpicxx \
    --with-fc=0 \
    --with-debugging=0 \
    --with-shared-libraries=1 \
    --download-fblaslapack \
    --download-mumps \
    --download-scalapack \
    --download-parmetis \
    --download-metis \
    COPTFLAGS='-O3 -march=native' \
    CXXOPTFLAGS='-O3 -march=native'

# Build PETSc
echo ""
echo "Building PETSc (this will take 10-20 minutes)..."
make PETSC_DIR="$PETSC_INSTALL_DIR" PETSC_ARCH="$PETSC_ARCH" all -j"$NPROC"

# Install petsc4py
echo ""
echo "Installing petsc4py..."
export PETSC_DIR="$PETSC_INSTALL_DIR"
export PETSC_ARCH="$PETSC_ARCH"

python3 -m pip install --no-cache-dir "petsc4py==$PETSC_VERSION"

# Create environment setup script
ENV_SCRIPT="$PETSC_INSTALL_DIR/petsc_env.sh"
cat > "$ENV_SCRIPT" << EOF
# Source this file to set up PETSc environment
# Usage: source $ENV_SCRIPT
export PETSC_DIR="$PETSC_INSTALL_DIR"
export PETSC_ARCH="$PETSC_ARCH"
EOF

echo ""
echo "=============================================="
echo "PETSc installation complete!"
echo "=============================================="
echo ""
echo "IMPORTANT: Set these environment variables before using GaPFlow with 2D FEM:"
echo ""
echo "  export PETSC_DIR=\"$PETSC_INSTALL_DIR\""
echo "  export PETSC_ARCH=\"$PETSC_ARCH\""
echo ""
echo "Or source the setup script:"
echo ""
echo "  source $ENV_SCRIPT"
echo ""
echo "Add the export lines to your ~/.bashrc for persistence."
echo ""

# Verify installation
echo "Verifying installation..."
cd "$(dirname "$0")"
python3 .check_petsc.py
