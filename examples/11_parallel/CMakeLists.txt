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
        NAME example_thread_teams_cc
        SOURCES
        thread_teams.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_thread_teams_lambda_cc
        SOURCES
        thread_teams_lambda.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_nested_parallel_for_cc
        SOURCES
        nested_parallel_for.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_vectorization_cc
        SOURCES
        vectorization.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_team_scan_cc
        SOURCES
        team_scan.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)

carbin_cc_binary(
        NAME example_launch_bounds_reduce_cc
        SOURCES
        launch_bounds_reduce.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        flare::core
)
