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

set(SIMD_TESTS_SRC
        api_test.cc
        arch_test.cc
        basic_math_test.cc
        batch_test.cc
        batch_bool_test.cc
        batch_cast_test.cc
        batch_complex_test.cc
        batch_float_test.cc
        batch_int_test.cc
        bitwise_cast_test.cc
        batch_constant_test.cc
        batch_manip_test.cc
        complex_exponential_test.cc
        complex_hyperbolic_test.cc
        complex_power_test.cc
        complex_trigonometric_test.cc
        conversion_test.cc
        custom_default_arch_test.cc
        error_gamma_test.cc
        explicit_batch_instantiation_test.cc
        exponential_test.cc
        extract_pair_test.cc
        fp_manipulation_test.cc
        hyperbolic_test.cc
        load_store_test.cc
        memory_test.cc
        poly_evaluation_test.cc
        power_test.cc
        rounding_test.cc
        select_test.cc
        shuffle_test.cc
        sum_test.cc
        traits_test.cc
        trigonometric_test.cc
        simd_api_test.cc
)

carbin_cc_test(
        NAME simd_test
        SOURCES
        ${SIMD_TESTS_SRC}
        simd_test_main.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::core
)
