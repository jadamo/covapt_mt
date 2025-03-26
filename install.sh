#!/bin/bash

# exit as soon as a command fails:
set -e

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Detect the operating system (they require slightly different dependencies)
OS=$(uname -s)
ARCH=$(uname -m)

echo "Operating system: $OS"
echo "Architecture: $ARCH"

# NOTE: This is only for when nbodykit is used in covapt_mt. 
# Change this when switching to DESI code.
if [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    PLATFORM=linux-64
elif [[ "$OS" == "Darwin" ]]; then
    PLATFORM=osx-64
else
    echo "ERROR: Unsupported platform detected! Installation failed!"
    exit 1
fi

echo "Setting PLATFORM=$PLATFORM"

########### conda stuff

# Always execute this script with bash, so that conda shell.hook works.
# Relevant conda bug: https://github.com/conda/conda/issues/7980
if [[ -z "$BASH_VERSION" ]];
then
    exec bash "$0" "$@"
fi

eval "$(conda shell.bash hook)"

conda env create -n covapt --platform $PLATFORM -f "$SCRIPT_DIR/environment.yaml"

conda activate covapt

python -m pip install -e "$SCRIPT_DIR"

conda deactivate

echo "Succesfully installed covapt env!"
