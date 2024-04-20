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
function(generate_product_version outfile)
  set(options)
  set(oneValueArgs
    COMPANY_NAME
    FILE_DESCRIPTION
    FILE_NAME
    ORIGINAL_FILE_NAME
    COMPANY_COPYRIGHT
  )
  set(multiValueArgs)
  cmake_parse_arguments(PRODUCT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT PRODUCT_COMPANY_NAME OR "${PRODUCT_COMPANY_NAME}" STREQUAL "")
      set(PRODUCT_COMPANY_NAME "Flare")
  endif()
  if(NOT PRODUCT_FILE_DESCRIPTION OR "${PRODUCT_FILE_DESCRIPTION}" STREQUAL "")
    set(PRODUCT_FILE_DESCRIPTION "Flare Library")
  endif()
  if(NOT PRODUCT_FILE_NAME OR "${PRODUCT_FILE_NAME}" STREQUAL "")
    set(PRODUCT_FILE_NAME "${PROJECT_NAME}")
  endif()
  if(NOT PRODUCT_ORIGINAL_FILE_NAME OR "${PRODUCT_ORIGINAL_FILE_NAME}" STREQUAL "")
    set(PRODUCT_ORIGINAL_FILE_NAME "${PRODUCT_FILE_NAME}")
  endif()
  if(NOT PRODUCT_FILE_DESCRIPTION OR "${PRODUCT_FILE_DESCRIPTION}" STREQUAL "")
      set(PRODUCT_FILE_DESCRIPTION "${PRODUCT_FILE_NAME}")
  endif()
  if(NOT PRODUCT_COMPANY_COPYRIGHT OR "${PRODUCT_COMPANY_COPYRIGHT}" STREQUAL "")
    string(TIMESTAMP PRODUCT_CURRENT_YEAR "%Y")
    set(PRODUCT_COMPANY_COPYRIGHT "${PRODUCT_COMPANY_NAME} (C) Copyright ${PRODUCT_CURRENT_YEAR}")
  endif()

  set(PRODUCT_VERSION ${PROJECT_VERSION})
  set(PRODUCT_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set(PRODUCT_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set(PRODUCT_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  set(PRODUCT_INTERNAL_FILE_NAME ${PRODUCT_ORIGINAL_FILE_NAME})

  set(ver_res_file "${PROJECT_BINARY_DIR}/${PRODUCT_FILE_NAME}_version_info.rc")
  configure_file(
    ${PROJECT_SOURCE_DIR}/CMakeModules/version_info.rc.in
    ${ver_res_file}
  )
  set(${outfile} ${ver_res_file} PARENT_SCOPE)
endfunction()
