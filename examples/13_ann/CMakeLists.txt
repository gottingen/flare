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

carbin_cc_binary(
        NAME example_distance_l1_cc
        SOURCES
        distance_l1.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_distance_l1_cu
        SOURCES
        distance_l1.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_norm_l1_cc
        SOURCES
        norm_l1.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_norm_l1_cu
        SOURCES
        norm_l1.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_distance_l2_cc
        SOURCES
        distance_l2.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_norm_l2_cc
        SOURCES
        norm_l2.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_distance_ip_cc
        SOURCES
        distance_ip.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_distance_jaccard_cc
        SOURCES
        distance_jaccard.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_distance_hamming_cc
        SOURCES
        distance_hamming.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)
carbin_cc_binary(
        NAME example_parallel_distance_l2_openmp
        SOURCES
        parallel_distance_l2_openmp.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_parallel_distance_l2_batch
        SOURCES
        parallel_distance_l2_batch.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_distance_l2_batch
        SOURCES
        distance_l2_batch.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)
