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

set(BUILD_OFFLINE ON)
# Turn ON disconnected flag when connected to cloud
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL
    "Disable Download/Update stages of FetchContent workflow" FORCE)

message(STATUS "No cloud connection. Attempting offline build if dependencies are available")

# Track dependencies download persistently across multiple
# cmake configure runs. *_POPULATED variables are reset for each
# cmake run to 0. Hence, this internal cache value is needed to
# check for already (from previous cmake run's) populated data
# during the current cmake run if it looses network connection.
set(FLY_INTERNAL_DOWNLOAD_FLAG OFF CACHE BOOL "Deps Download Flag")

# Override fetch content base dir before including fly_fetch_content
set(FETCHCONTENT_BASE_DIR "${flare_BINARY_DIR}/extern" CACHE PATH
    "Base directory where Flare dependencies are downloaded and/or built" FORCE)

include(fly_fetch_content)

mark_as_advanced(
  FLY_INTERNAL_DOWNLOAD_FLAG
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_QUIET
  FETCHCONTENT_FULLY_DISCONNECTED
  FETCHCONTENT_UPDATES_DISCONNECTED
)

macro(set_and_mark_depnames_advncd var name)
  string(TOLOWER ${name} ${var})
  string(TOUPPER ${name} ${var}_ucname)
  mark_as_advanced(
      FETCHCONTENT_SOURCE_DIR_${${var}_ucname}
      FETCHCONTENT_UPDATES_DISCONNECTED_${${var}_ucname}
  )
endmacro()

set_and_mark_depnames_advncd(assets_prefix "af_assets")
set_and_mark_depnames_advncd(testdata_prefix "af_test_data")
set_and_mark_depnames_advncd(glad_prefix "fly_glad")
set_and_mark_depnames_advncd(threads_prefix "af_threads")
set_and_mark_depnames_advncd(cub_prefix "nv_cub")
set_and_mark_depnames_advncd(cl2hpp_prefix "ocl_cl2hpp")
set_and_mark_depnames_advncd(clblast_prefix "ocl_clblast")
set_and_mark_depnames_advncd(clfft_prefix "ocl_clfft")
set_and_mark_depnames_advncd(boost_prefix "boost_compute")

macro(fly_dep_check_and_populate dep_prefix)
  set(single_args URI REF)
  cmake_parse_arguments(adcp_args "" "${single_args}" "" ${ARGN})

  if("${adcp_args_URI}" STREQUAL "")
    message(FATAL_ERROR [=[
        Cannot check requested dependency source's availability.
        Please provide a valid URI(almost always a URL to a github repo).
        Note that the above error message if for developers of Flare.
        ]=])
  endif()

  string(FIND "${adcp_args_REF}" "=" adcp_has_algo_id)

  if(${BUILD_OFFLINE} AND NOT ${FLY_INTERNAL_DOWNLOAD_FLAG})
    if(NOT ${adcp_has_algo_id} EQUAL -1)
      FetchContent_Populate(${dep_prefix}
        QUIET
        URL            ${adcp_args_URI}
        URL_HASH       ${adcp_args_REF}
        DOWNLOAD_COMMAND \"\"
        UPDATE_DISCONNECTED ON
        SOURCE_DIR     "${flare_SOURCE_DIR}/extern/${dep_prefix}-src"
        BINARY_DIR     "${flare_BINARY_DIR}/extern/${dep_prefix}-build"
        SUBBUILD_DIR   "${flare_BINARY_DIR}/extern/${dep_prefix}-subbuild"
      )
    elseif("${adcp_args_REF}" STREQUAL "")
      FetchContent_Populate(${dep_prefix}
        QUIET
        URL            ${adcp_args_URI}
        DOWNLOAD_COMMAND \"\"
        UPDATE_DISCONNECTED ON
        SOURCE_DIR     "${flare_SOURCE_DIR}/extern/${dep_prefix}-src"
        BINARY_DIR     "${flare_BINARY_DIR}/extern/${dep_prefix}-build"
        SUBBUILD_DIR   "${flare_BINARY_DIR}/extern/${dep_prefix}-subbuild"
      )
    else()
      # The left over alternative is assumed to be a cloud hosted git repository
      FetchContent_Populate(${dep_prefix}
        QUIET
        GIT_REPOSITORY ${adcp_args_URI}
        GIT_TAG        ${adcp_args_REF}
        DOWNLOAD_COMMAND \"\"
        UPDATE_DISCONNECTED ON
        SOURCE_DIR     "${flare_SOURCE_DIR}/extern/${dep_prefix}-src"
        BINARY_DIR     "${flare_BINARY_DIR}/extern/${dep_prefix}-build"
        SUBBUILD_DIR   "${flare_BINARY_DIR}/extern/${dep_prefix}-subbuild"
      )
    endif()
  else()
    if(NOT ${adcp_has_algo_id} EQUAL -1)
      FetchContent_Declare(${dep_prefix}
        URL            ${adcp_args_URI}
        URL_HASH       ${adcp_args_REF}
      )
    elseif("${adcp_args_REF}" STREQUAL "")
      FetchContent_Declare(${dep_prefix}
        URL            ${adcp_args_URI}
      )
    else()
      # The left over alternative is assumed to be a cloud hosted git repository
      FetchContent_Declare(${dep_prefix}
        GIT_REPOSITORY ${adcp_args_URI}
        GIT_TAG        ${adcp_args_REF}
      )
    endif()
    FetchContent_GetProperties(${dep_prefix})
    if(NOT ${dep_prefix}_POPULATED)
      FetchContent_Populate(${dep_prefix})
    endif()
    set(FLY_INTERNAL_DOWNLOAD_FLAG ON CACHE BOOL "Deps Download Flag" FORCE)
  endif()
endmacro()
