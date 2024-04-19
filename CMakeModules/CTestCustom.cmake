#
# Copyright 2023 The EA Authors.
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
#

set(CTEST_CUSTOM_ERROR_POST_CONTEXT 200)
set(CTEST_CUSTOM_ERROR_PRE_CONTEXT 200)
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS 300)
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS 300)

if(WIN32)
  if(CMAKE_GENERATOR MATCHES "Ninja")
    set(CTEST_CUSTOM_POST_TEST ./bin/print_info.exe)
  endif()
else()
  set(CTEST_CUSTOM_POST_TEST ./test/print_info)
endif()

list(APPEND CTEST_CUSTOM_COVERAGE_EXCLUDE
  "test"

  # All external and third_party libraries
  "extern/.*"
  "test/mmio/.*"
  "src/backend/cpu/threads/.*"
  "src/backend/cuda/cub/.*"
  "cl2.hpp"

  # Remove bin2cpp from coverage
  "CMakeModules/.*")
