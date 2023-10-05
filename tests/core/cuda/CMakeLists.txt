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


carbin_cc_test(
        NAME cuda_range_test
        SOURCES
        mdrange_a_test.cu
        mdrange_b_test.cu
        mdrange_c_test.cu
        mdrange_d_test.cu
        mdrange_e_test.cu
        mdrange_f_test.cu
        mdrange_g_test.cu
        mdrange_policy_constructors_test.cu
        mdrange_reduce_test.cu
        mdspan_test.cu
        min_max_clamp_test.cu
        numeric_traits_test.cu
        other_test.cu
        parallel_scan_range_policy_test.cu
        printf_test.cu
        quad_precision_math_test.cu
        range_policy_constructors_test.cu
        range_policy_require_test.cu
        range_policy_test.cu
        reducer_ctads_test.cu
        reducers_a_test.cu
        reducers_b_test.cu
        reducers_c_test.cu
        reducers_d_test.cu
        reducers_e_test.cu
        reductions_device_view_test.cu
        reductions_test.cu
        shared_alloc_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME cuda_view_api_test
        SOURCES
        view_64bit_test.cu
        view_api_a_test.cu
        view_api_b_test.cu
        view_api_c_test.cu
        view_api_d_test.cu
        view_api_e_test.cu
        view_copy_a_test.cu
        view_copy_b_test.cu
        view_hooks_test.cu
        #view_layout_stride_assignment_test.cu
        view_mapping_a_test.cu
        view_mapping_b_test.cu
        view_mapping_subview_test.cu
        #view_memory_access_violation_test.cu
        view_of_class_test.cu
        view_resize_test.cu
        view_subview_test.cu
        without_initializing_test.cu
        work_graph_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME cuda_subview_test
        SOURCES
        subview_01_test.cu
        subview_02_test.cu
        subview_03_test.cu
        subview_04_test.cu
        subview_05_test.cu
        subview_06_test.cu
        subview_07_test.cu
        subview_08_test.cu
        subview_09_test.cu
        subview_10_test.cu
        subview_11_test.cu
        subview_12_test.cu
        subview_13_test.cu
        subview_14_test.cu
        subview_a_test.cu
        subview_b_test.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)