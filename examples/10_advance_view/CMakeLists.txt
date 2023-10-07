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
        NAME example_data_layouts_cc
        SOURCES
        data_layouts.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_data_layouts_cu
        SOURCES
        data_layouts.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_memory_traits_cc
        SOURCES
        memory_traits.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_memory_traits_cu
        SOURCES
        memory_traits.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_subviews_cc
        SOURCES
        subviews.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_subviews_cu
        SOURCES
        subviews.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_dual_view_cc
        SOURCES
        dual_view.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_dual_view_cu
        SOURCES
        dual_view.cu
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        flare::core
)
