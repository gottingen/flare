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
        NAME core_serial_graph_test
        SOURCES
        serial_graph_test.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)

carbin_cc_test(
        NAME serial_atomic_operations_test
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
        atomic_view_test.cc
        ${TEST_MAIN_SRC}
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CUDA_CUDA_LIBRARY}
        flare::flare
)
