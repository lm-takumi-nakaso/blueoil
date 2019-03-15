#!/bin/bash

set -eo pipefail

NAME="$0"
ABS="$(cd "$(dirname "${NAME}")" && pwd)"

if [ -z "$(ls -A "${ABS}"lmnet/third_party/coco)" ]; then
    echo "seems submodule is not initialized correctly. Please execute following command, before re-run this script."
    echo "git submodule update --init --recursive"
    exit 1
fi

DOCKER_IMAGE="$(id -un)"_blueoil:local_build
docker build -t "${DOCKER_IMAGE}" --build-arg python_version="3.6.3" -f "${ABS}"/docker/Dockerfile "${ABS}"

echo '' > "${ABS}"/blueoil.env
echo BLUEOIL_CONTAINER_TYPE=singularity >> "${ABS}"/blueoil.env
echo BLUEOIL_CONTAINER_IMAGE="$(printf %q "${DOCKER_IMAGE}")" >> "${ABS}"/blueoil.env
