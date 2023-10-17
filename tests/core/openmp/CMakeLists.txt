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
        NAME core_openmp_inter_op_test
        SOURCES
        openmp_inter_op_test.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::core
)

carbin_cc_test(
        NAME openmp_atomic_operations_test
        SOURCES
        atomic_operations_complexdouble_test.cc
        atomic_operations_complexfloat_test.cc
        atomic_operations_double_test.cc
        atomic_operations_float_test.cc
        atomic_operations_int_test.cc
        atomic_operations_longint_test.cc
        atomic_operations_longlongint_test.cc
        atomic_operations_shared_test.cc
        atomic_operations_unsignedint_test.cc
        atomic_operations_unsignedlongint_test.cc
        atomic_test.cc
        atomic_tensor_test.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::core
)

carbin_cc_test(
        NAME openmp_misc_test
        SOURCES
        bit_manipulation_builtins_test.cc
        block_size_deduction_test.cc
        checked_integer_ops_test.cc
        common_policy_constructors_test.cc
        common_policy_interface_test.cc
        complex_test.cc
        concepts_test.cc
        crs_test.cc
        deep_copy_alignment_test.cc
        exec_space_partitioning_test.cc
        execution_space_test.cc
        functor_analysis_test.cc
        host_shared_ptr_access_on_device_test.cc
        host_shared_ptr_test.cc
        init_test.cc
        join_backward_compatibility_test.cc
        local_deep_copy_test.cc
        mathematical_constants_test.cc
        mathematical_functions1_test.cc
        mathematical_functions2_test.cc
        mathematical_functions3_test.cc
        mathematical_special_functions_test.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::core
)

carbin_cc_test(
        NAME openmp_range_test
        SOURCES
        mdrange_a_test.cc
        mdrange_b_test.cc
        mdrange_c_test.cc
        mdrange_d_test.cc
        mdrange_e_test.cc
        mdrange_f_test.cc
        mdrange_g_test.cc
        mdrange_policy_constructors_test.cc
        mdrange_reduce_test.cc
        mdspan_test.cc
        min_max_clamp_test.cc
        numeric_traits_test.cc
        other_test.cc
        parallel_scan_range_policy_test.cc
        printf_test.cc
        quad_precision_math_test.cc
        range_policy_constructors_test.cc
        range_policy_require_test.cc
        range_policy_test.cc
        reducer_ctads_test.cc
        reducers_a_test.cc
        reducers_b_test.cc
        reducers_c_test.cc
        reducers_d_test.cc
        reducers_e_test.cc
        reductions_device_tensor_test.cc
        reductions_test.cc
        shared_alloc_test.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::core
)


carbin_cc_test(
        NAME openmp_tensor_api_test
        SOURCES
        tensor_64bit_test.cc
        tensor_api_a_test.cc
        tensor_api_b_test.cc
        tensor_api_c_test.cc
        tensor_api_d_test.cc
        tensor_api_e_test.cc
        tensor_copy_a_test.cc
        tensor_copy_b_test.cc
        tensor_hooks_test.cc
        #tensor_layout_stride_assignment_test.cc
        tensor_mapping_a_test.cc
        tensor_mapping_b_test.cc
        tensor_mapping_subtensor_test.cc
        #tensor_memory_access_violation_test.cc
        tensor_of_class_test.cc
        tensor_resize_test.cc
        tensor_subtensor_test.cc
        without_initializing_test.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::core
)

carbin_cc_test(
        NAME openmp_subtensor_test
        SOURCES
        subtensor_01_test.cc
        subtensor_02_test.cc
        subtensor_03_test.cc
        subtensor_04_test.cc
        subtensor_05_test.cc
        subtensor_06_test.cc
        subtensor_07_test.cc
        subtensor_08_test.cc
        subtensor_09_test.cc
        subtensor_10_test.cc
        subtensor_11_test.cc
        subtensor_12_test.cc
        subtensor_13_test.cc
        subtensor_14_test.cc
        subtensor_a_test.cc
        subtensor_b_test.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::core
)
