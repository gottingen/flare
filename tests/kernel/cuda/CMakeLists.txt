#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


carbin_cc_test(
        NAME cuda_abs_test
        SOURCES
        cuda_abs_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS} "-Wno-unused-but-set-parameter"
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::blas
)
carbin_cc_test(
        NAME cuda_axpby_test
        SOURCES
        cuda_axpby_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS} "-Wno-unused-but-set-parameter"
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::blas
)
carbin_cc_test(
        NAME cuda_dot_test
        SOURCES
        cuda_dot_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS} "-Wno-unused-but-set-parameter"
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::blas
)

carbin_cc_test(
        NAME cuda_iamax_test
        SOURCES
        cuda_iamax_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS} "-Wno-unused-but-set-parameter"
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::blas
)

carbin_cc_test(
        NAME cuda_nrm1_test
        SOURCES
        cuda_nrm1_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS} "-Wno-unused-but-set-parameter"
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::blas
)
