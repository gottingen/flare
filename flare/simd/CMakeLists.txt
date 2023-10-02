#I have to leave these here for tribits
FLARE_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
FLARE_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

#-----------------------------------------------------------------------------

FILE(GLOB SIMD_HEADERS *.h)
FILE(GLOB SIMD_SOURCES *.cc)

INSTALL (
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${FLARE_HEADER_DIR}
  FILES_MATCHING PATTERN "*.hpp"
)

#-----------------------------------------------------------------------------

# We have to pass the sources in here for Tribits
# These will get ignored for standalone CMake and a true interface library made
FLARE_ADD_LIBRARY(
  flare_simd
  SOURCES ${SIMD_SOURCES}
  HEADERS ${SIMD_HEADERS}
)
FLARE_LIB_INCLUDE_DIRECTORIES(flare_simd
  ${FLARE_TOP_BUILD_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)
