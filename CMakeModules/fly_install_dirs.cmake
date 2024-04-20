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
# Sets Flare installation paths.
#

include(GNUInstallDirs)

# NOTE: These paths are all relative to the project installation prefix.

# Executables
if(NOT DEFINED FLY_INSTALL_BIN_DIR)
  set(FLY_INSTALL_BIN_DIR "lib" CACHE PATH "Installation path for executables")
endif()

# Libraries
if(NOT DEFINED FLY_INSTALL_LIB_DIR)
  if(WIN32)
    set(FLY_INSTALL_LIB_DIR "lib" CACHE PATH "Installation path for libraries")
  else()
    set(FLY_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Installation path for libraries")
  endif()
endif()

# Header files
if(NOT DEFINED FLY_INSTALL_INC_DIR)
  set(FLY_INSTALL_INC_DIR "include" CACHE PATH "Installation path for headers")
endif()

set(DATA_DIR "share/flare")

# Documentation
if(NOT DEFINED FLY_INSTALL_DOC_DIR)
  if (WIN32)
    set(FLY_INSTALL_DOC_DIR "doc" CACHE PATH "Installation path for documentation")
  else ()
      set(FLY_INSTALL_DOC_DIR "${DATA_DIR}/doc" CACHE PATH "Installation path for documentation")
  endif ()
endif()

if(NOT DEFINED FLY_INSTALL_EXAMPLE_DIR)
  if (WIN32)
    set(FLY_INSTALL_EXAMPLE_DIR "examples" CACHE PATH "Installation path for examples")
  else ()
    set(FLY_INSTALL_EXAMPLE_DIR "${DATA_DIR}/examples" CACHE PATH "Installation path for examples")
  endif ()
endif()

if(NOT DEFINED FLY_INSTALL_ASSERTS_DIR)
  if (WIN32)
    set(FLY_INSTALL_ASSERTS_DIR "assert" CACHE PATH "Installation path for examples")
  else ()
    set(FLY_INSTALL_ASSERTS_DIR "${DATA_DIR}/asserts" CACHE PATH "Installation path for examples")
  endif ()
endif()

# Man pages
if(NOT DEFINED FLY_INSTALL_MAN_DIR)
    set(FLY_INSTALL_MAN_DIR "${DATA_DIR}/man" CACHE PATH "Installation path for man pages")
endif()

# CMake files
if(NOT DEFINED FLY_INSTALL_CMAKE_DIR)
  if (WIN32)
    set(FLY_INSTALL_CMAKE_DIR "cmake" CACHE PATH "Installation path for CMake files")
  else ()
    set(FLY_INSTALL_CMAKE_DIR "${FLY_INSTALL_LIB_DIR}/cmake/flare" CACHE PATH "Installation path for CMake files")
  endif ()
endif()

mark_as_advanced(
  FLY_INSTALL_BIN_DIR
  FLY_INSTALL_LIB_DIR
  FLY_INSTALL_INC_DIR
  FLY_INSTALL_DATA_DIR
  FLY_INSTALL_DOC_DIR
  FLY_INSTALL_EXAMPLE_DIR
  FLY_INSTALL_MAN_DIR
  FLY_INSTALL_CMAKE_DIR)
