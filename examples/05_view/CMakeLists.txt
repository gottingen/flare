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
        NAME example_simple_view_cc
        SOURCES
        simple_view.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_simple_view_cu
        SOURCES
        simple_view.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_simple_view_lambda_cc
        SOURCES
        simple_view_lambda.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_simple_view_lambda_cu
        SOURCES
        simple_view_lambda.cu
        COPTS ${CARBIN_CXX_OPTIONS}
        CUOPTS ${CARBIN_CU_OPTIONS}
        DEPS
        flare::core
)