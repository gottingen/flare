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

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cpp" FILESTRINGS)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_BINARY_OPS")
        string(REPLACE "// SBK_BINARY_OPS:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_BINARY_OPS ${TEMP})
    endif()
endforeach()

foreach(SBK_BINARY_OP ${SBK_BINARY_OPS})
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/scan_by_key/scan_by_key_impl.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/kernel/scan_by_key/scan_by_key_impl_${SBK_BINARY_OP}.cpp"
  )

  list(
    APPEND
    scan_by_key_sources
    "${CMAKE_CURRENT_BINARY_DIR}/kernel/scan_by_key/scan_by_key_impl_${SBK_BINARY_OP}.cpp"
  )
endforeach(SBK_BINARY_OP ${SBK_BINARY_OPS})
