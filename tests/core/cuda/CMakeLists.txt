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
        NAME core_cuda_timing_based_test
        SOURCES
        cuda_debug_serial_execution_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME core_cuda_pinuvm_space_test
        SOURCES
        cuda_debug_pinuvm_space_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME core_cuda_graph_test
        SOURCES
        cuda_graph_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME core_cuda_inter_op_init_test
        SOURCES
        cuda_inter_op_init_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME core_cuda_inter_op_streams_test
        SOURCES
        cuda_inter_op_streams_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME cuda_reducer_view_size_limit_test
        SOURCES
        cuda_reducer_view_size_limit_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME core_cuda_spaces_test
        SOURCES
        cuda_spaces_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME cuda_team_scratch_streams_test
        SOURCES
        cuda_team_scratch_streams_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)