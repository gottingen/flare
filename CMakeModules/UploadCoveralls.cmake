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

IF(${WITH_COVERAGE})
    Find_Program(COVERALLS_EXECUTABLE coveralls)

    #FIXME: Remove after eddyxu/cpp-coveralls #75 accepted
    SET(COVERALL_ARGS "--include src --include include --exclude src/backend/opencl/cl.hpp --exclude test --gcov-options '\\-lp'")

    ADD_CUSTOM_TARGET(coveralls
        COMMAND ${COVERALLS_EXECUTABLE}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Creating coverage"
        )
ENDIF(${WITH_COVERAGE})
