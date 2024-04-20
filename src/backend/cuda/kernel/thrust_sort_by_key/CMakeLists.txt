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

file(STRINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl.cu"
    FILESTRINGS)

foreach(STR ${FILESTRINGS})
    if(${STR} MATCHES "// SBK_TYPES")
        string(REPLACE "// SBK_TYPES:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_TYPES ${TEMP})
    elseif(${STR} MATCHES "// SBK_INSTS:")
        string(REPLACE "// SBK_INSTS:" "" TEMP ${STR})
        string(REPLACE " " ";" SBK_INSTS ${TEMP})
    endif()
endforeach()

foreach(SBK_TYPE ${SBK_TYPES})
  foreach(SBK_INST ${SBK_INSTS})
    set(INSTANTIATESBK_INST "INSTANTIATE${SBK_INST}")

    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl_${SBK_TYPE}_${SBK_INST}.cu"
    )

    list(
      APPEND
      thrust_sort_sources
      "${CMAKE_CURRENT_BINARY_DIR}/kernel/thrust_sort_by_key/thrust_sort_by_key_impl_${SBK_TYPE}_${SBK_INST}.cu"
    )
  endforeach(SBK_INST ${SBK_INSTS})
endforeach(SBK_TYPE ${SBK_TYPES})
