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
message(STATUS "Using VCPKG FindLAPACK from package 'lapack-reference'")
set(LAPACK_PREV_MODULE_PATH ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

list(REMOVE_ITEM ARGS "NO_MODULE")
list(REMOVE_ITEM ARGS "CONFIG")
list(REMOVE_ITEM ARGS "MODULE")

_find_package(${ARGS})

set(CMAKE_MODULE_PATH ${LAPACK_PREV_MODULE_PATH})
