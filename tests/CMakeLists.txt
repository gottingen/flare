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

#[[carbin_cc_test(
        NAME custom_comparator_test
        SOURCES custom_comparator_test.cc
        DEPS turbo::turbo ${CARBIN_DEPS_LINK} ${GTEST_LIB} ${GTEST_MAIN_LIB}
        COPTS ${USER_CXX_FLAGS}
)]]
#[[
carbin_cc_test(
        NAME foo_test
        SOURCES foo_test.cc
        DEPS changeme::foo ${CARBIN_DEPS_LINK} gtest gtest_main
        COPTS ${USER_CXX_FLAGS}
)

carbin_cc_test(
        NAME foo_doctest
        SOURCES foo_doctest.cc
        DEPS changeme::foo ${CARBIN_DEPS_LINK} doctest
        COPTS ${USER_CXX_FLAGS}
)
]]
include_directories(category_files)
include_directories(${PROJECT_SOURCE_DIR}/tests/doctest)
include_directories(core/includes)
include_directories(algorithm)
include_directories(container)
include_directories(${PROJECT_SOURCE_DIR}/tests)
set(TEST_MAIN_SRC
        ${PROJECT_SOURCE_DIR}/tests/core/unit_test_main.cc
)

if (FLARE_ENABLE_CORE_TEST)
    add_subdirectory(algorithm)
    add_subdirectory(container)
    add_subdirectory(compile_only)
    add_subdirectory(core)
endif ()

if (FLARE_ENABLE_KERNEL_TEST)
    add_subdirectory(kernel)
endif ()

add_subdirectory(simd)
