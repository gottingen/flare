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

set(DATA_DIR "share/Flare")

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

# Man pages
if(NOT DEFINED FLY_INSTALL_MAN_DIR)
    set(FLY_INSTALL_MAN_DIR "${DATA_DIR}/man" CACHE PATH "Installation path for man pages")
endif()

# CMake files
if(NOT DEFINED FLY_INSTALL_CMAKE_DIR)
  if (WIN32)
    set(FLY_INSTALL_CMAKE_DIR "cmake" CACHE PATH "Installation path for CMake files")
  else ()
    set(FLY_INSTALL_CMAKE_DIR "${DATA_DIR}/cmake" CACHE PATH "Installation path for CMake files")
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
