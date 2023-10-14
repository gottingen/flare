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

SET(COMPILE_ONLY_SOURCES
        array_test.cc
        create_mirror_test.cc
        detection_idiom_test.cc
        bit_manipulation_test.cc
        inter_op_test.cc
        string_manipulation_test.cc
        version_macros_test.cc
        tensor_rank_test.cc
        tensor_type_traits_test.cc
        type_list_test.cc
        extents_datatype_conversion_test.cc
)

carbin_cc_binary(
        NAME compile_only_test
        SOURCES
        ${COMPILE_ONLY_SOURCES}
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)
