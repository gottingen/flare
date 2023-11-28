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

#[[
carbin_cc_test(
        NAME tf_cuda_algorithm_updates_test
        SOURCES
        cuda_algorithm_updates_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_algorithms_test
        SOURCES
        cuda_algorithms_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)]]

carbin_cc_test(
        NAME tf_cuda_basic_updates_test
        SOURCES
        cuda_basic_updates_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_basics_test
        SOURCES
        cuda_basics_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_capturer_optimizer_test
        SOURCES
        cuda_capturer_optimizer_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_find_test
        SOURCES
        cuda_find_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_for_each_index_test
        SOURCES
        cuda_for_each_index_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_for_each_test
        SOURCES
        cuda_for_each_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_kmeans_test
        SOURCES
        cuda_kmeans_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_matrix_test
        SOURCES
        cuda_matrix_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_memory_test
        SOURCES
        cuda_memory_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_merge_test
        SOURCES
        cuda_merge_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_min_max_element_test
        SOURCES
        cuda_min_max_element_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_objects_test
        SOURCES
        cuda_objects_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_reduce_test
        SOURCES
        cuda_reduce_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_scan_test
        SOURCES
        cuda_scan_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)

carbin_cc_test(
        NAME tf_cuda_transform_test
        SOURCES
        cuda_transform_test.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        flare::core
)