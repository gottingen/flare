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

carbin_cc_test(
        NAME cuda_atomic_operations_test
        SOURCES
        atomic_operations_complexdouble_test.cu
        atomic_operations_complexfloat_test.cu
        atomic_operations_double_test.cu
        atomic_operations_float_test.cu
        atomic_operations_int_test.cu
        atomic_operations_longint_test.cu
        atomic_operations_longlongint_test.cu
        atomic_operations_shared_test.cu
        atomic_operations_unsignedint_test.cu
        atomic_operations_unsignedlongint_test.cu
        atomic_test.cu
        atomic_view_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME cuda_misc_test
        SOURCES
        bit_manipulation_builtins_test.cu
        block_size_deduction_test.cu
        checked_integer_ops_test.cu
        common_policy_constructors_test.cu
        common_policy_interface_test.cu
        complex_test.cu
        concepts_test.cu
        crs_test.cu
        deep_copy_alignment_test.cu
        exec_space_partitioning_test.cu
        execution_space_test.cu
        functor_analysis_test.cu
        host_shared_ptr_access_on_device_test.cu
        host_shared_ptr_test.cu
        init_test.cu
        join_backward_compatibility_test.cu
        local_deep_copy_test.cu
        mathematical_constants_test.cu
        mathematical_functions1_test.cu
        mathematical_functions2_test.cu
        mathematical_functions3_test.cu
        mathematical_special_functions_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

