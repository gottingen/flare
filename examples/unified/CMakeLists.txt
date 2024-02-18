# Copyright (c) 2018, Flare
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.5)
project(Flare-Example-Unified
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(Flare)

if(Flare_Unified_FOUND)
  # Simple unified backend example
  add_executable(basic_unified basic.cpp)
  target_link_libraries(basic_unified Flare::fly)
endif()
