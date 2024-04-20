#! /usr/bin/env bash
# Copyright 2023 The titan-search Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


set -xue
set -o pipefail

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/../
DIST_PATH=${ROOT_PATH}/dist
BUILD_PATH=${ROOT_PATH}/build

# mkdir lib
if [ -d "${BUILD_PATH}" ]; then
  rm -rf "${BUILD_PATH}"
fi
if [ -d "${DIST_PATH}" ]; then
  rm -rf "${DIST_PATH}"
fi

mkdir "${DIST_PATH}"
mkdir -p "${BUILD_PATH}"
# build cuda
cmake -S ${ROOT_PATH} -B ${BUILD_PATH} -DBUILD_TESTING=OFF -DFLY_BUILD_EXAMPLES=OFF
cmake --build ${BUILD_PATH} -j 4
cmake --build ${BUILD_PATH} --target package
cp ${BUILD_PATH}/package/flare*.sh ${DIST_PATH}

# build n ocuda
rm -rf ${BUILD_PATH}/*
cmake -S ${ROOT_PATH} -B ${BUILD_PATH} -DBUILD_TESTING=OFF -DFLY_BUILD_EXAMPLES=OFF -DBUILD_CUDA=OFF
cmake --build ${BUILD_PATH} -j 4
cmake --build ${BUILD_PATH} --target package
cp ${BUILD_PATH}/package/flare*.sh ${DIST_PATH}



