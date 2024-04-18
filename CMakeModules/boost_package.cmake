# Copyright (c) 2017, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

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
