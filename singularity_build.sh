#!/bin/bash

set -eo pipefail

NAME="$0"
ABS="$(cd "$(dirname "${NAME}")" && pwd)"

if [ -z "$(ls -A "${ABS}"/lmnet/third_party/coco)" ]; then
    echo "seems submodule is not initialized correctly. Please execute following command, before re-run this script."
    echo "git submodule update --init --recursive"
    exit 1
fi

IMAGE_VERSION="$(cat "${ABS}"/setup.cfg | grep '^version = ' | sed 's/^[ \t\n]*version[ \t\n]*=[ \t\n]*//' | sed 's/[ \t\n]*$//')"
DOCKERHUB_IMAGE=docker://lmtakuminakaso/blueoil:"${IMAGE_VERSION}"
BUILD_DIR=build
mkdir -p "${ABS}"/"${BUILD_DIR}"
SINGULARITY_IMAGE_NAME=blueoil-"${IMAGE_VERSION}".simg
SINGULARITY_IMAGE="${BUILD_DIR}"/"${SINGULARITY_IMAGE_NAME}"
singularity build "${ABS}"/"${SINGULARITY_IMAGE}" "${DOCKERHUB_IMAGE}"

ENV_SH="${BUILD_DIR}"/env.sh
echo -n '' > "${ABS}"/"${ENV_SH}"
echo 'BLUEOIL_CONTAINER_TYPE="${BLUEOIL_CONTAINER_TYPE:-singularity}"' >> "${ABS}"/"${ENV_SH}"
echo 'BLUEOIL_CONTAINER_IMAGE="${BLUEOIL_CONTAINER_IMAGE:-"$(cd "$(dirname "$0")"; pwd)"/'"$(printf %q "${SINGULARITY_IMAGE}")"'}"' >> "${ABS}"/"${ENV_SH}"
