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

set(Boost_MIN_VER 107000)
set(Boost_MIN_VER_STR "1.70")

if(TARGET Boost::boost)
  set(BOOST_DEFINITIONS "BOOST_CHRONO_HEADER_ONLY;BOOST_COMPUTE_THREAD_SAFE;BOOST_COMPUTE_HAVE_THREAD_LOCAL")

  # NOTE: Basic and Windows options do not requre flags or libraries for
  #       backtraces
  if(FLY_STACKTRACE_TYPE STREQUAL "libbacktrace")
    list(APPEND BOOST_DEFINITIONS "BOOST_STACKTRACE_USE_BACKTRACE")
    set_target_properties(Boost::boost PROPERTIES
      INTERFACE_LINK_LIBRARIES ${Backtrace_LIBRARY})
  elseif(FLY_STACKTRACE_TYPE STREQUAL "addr2line")
    list(APPEND BOOST_DEFINITIONS "BOOST_STACKTRACE_USE_ADDR2LINE")
  elseif(FLY_STACKTRACE_TYPE STREQUAL "None")
      list(APPEND BOOST_DEFINITIONS "BOOST_STACKTRACE_USE_NOOP")
  endif()

  if(NOT FLY_STACKTRACE_TYPE STREQUAL "None" AND APPLE)
      list(APPEND BOOST_DEFINITIONS "BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED")
  endif()

  # NOTE: BOOST_CHRONO_HEADER_ONLY is required for Windows because otherwise it
  # will try to link with libboost-chrono.
  set_target_properties(Boost::boost PROPERTIES INTERFACE_COMPILE_DEFINITIONS
      "${BOOST_DEFINITIONS}")
endif()
