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
IF(NOT DEFINED MINBUILDTIME_FLAG)
    SET(MINBUILDTIME_FLAG OFF CACHE INTERNAL "Flag" FORCE)
ENDIF()

IF(${MIN_BUILD_TIME})
    IF(NOT ${CMAKE_BUILD_TYPE} MATCHES "Release")
        MESSAGE(WARNING "The MIN_BUILD_TIME Flag only works with Release.\
                        Other CMAKE_BUILD_TYPEs will be ignore this flag")
    ELSEIF(NOT ${MINBUILDTIME_FLAG})
    # BUILD_TYPE is Release - Set the flags
    # The flags should be set only when going from OFF -> ON. This is
    # determined by MINBUILDTIME_FLAG
    # IF FLAG is ON, then the flags were already set, no need to set them again
    # IF FLAG is OFF, then the flags are not set, so set them now, and back up
    # release flags
    MESSAGE(STATUS "MIN_BUILD_TIME: Setting Release flags to no optimizations")

        # Backup Default Release Flags
        SET(CMAKE_CXX_FLAGS_RELEASE_DEFAULT ${CMAKE_CXX_FLAGS_RELEASE} CACHE
            INTERNAL "Default compiler flags during release build" FORCE)
        SET(CMAKE_C_FLAGS_RELEASE_DEFAULT ${CMAKE_C_FLAGS_RELEASE} CACHE
            INTERNAL "Default compiler flags during release build" FORCE)
        SET(CMAKE_EXE_LINKER_FLAGS_RELEASE_DEFAULT ${CMAKE_EXE_LINKER_FLAGS_RELEASE} CACHE
            INTERNAL "Default linker flags during release build" FORCE)
        SET(CMAKE_MODULE_LINKER_FLAGS_RELEASE_DEFAULT ${CMAKE_MODULE_LINKER_FLAGS_RELEASE} CACHE
            INTERNAL "Default linker flags during release build" FORCE)
        SET(CMAKE_STATIC_LINKER_FLAGS_RELEASE_DEFAULT ${CMAKE_STATIC_LINKER_FLAGS_RELEASE} CACHE
            INTERNAL "Default linker flags during release build" FORCE)
        SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE_DEFAULT ${CMAKE_SHARED_LINKER_FLAGS_RELEASE} CACHE
            INTERNAL "Default linker flags during release build" FORCE)

        IF(MSVC)
            SET(CMAKE_CXX_FLAGS_RELEASE "/MD /Od /Ob1 /D NDEBUG" CACHE
                STRING "Flags used by the compiler during release builds." FORCE)
            SET(CMAKE_C_FLAGS_RELEASE "/MD /Od /Ob1 /D NDEBUG" CACHE
                STRING "Flags used by the compiler during release builds." FORCE)
            SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO" CACHE
                STRING "Flags used by the linker during release builds." FORCE)
            SET(CMAKE_MODULE_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO" CACHE
                STRING "Flags used by the linker during release builds." FORCE)
            SET(CMAKE_STATIC_LINKER_FLAGS_RELEASE "" CACHE
                STRING "Flags used by the linker during release builds." FORCE)
            SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO" CACHE
                STRING "Flags used by the linker during release builds." FORCE)
        ELSE(MSVC)
            SET(CMAKE_CXX_FLAGS_RELEASE "-O0 -DNDEBUG" CACHE
                STRING "Flags used by the compiler during release builds." FORCE)
            SET(CMAKE_C_FLAGS_RELEASE "-O0 -DNDEBUG" CACHE
                STRING "Flags used by the compiler during release builds." FORCE)
            SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "" CACHE
                STRING "Flags used by the linker during release builds." FORCE)
            SET(CMAKE_MODULE_LINKER_FLAGS_RELEASE "" CACHE
                STRING "Flags used by the linker during release builds." FORCE)
            SET(CMAKE_STATIC_LINKER_FLAGS_RELEASE "" CACHE
                STRING "Flags used by the linker during release builds." FORCE)
            SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE "" CACHE
                STRING "Flags used by the linker during release builds." FORCE)
        ENDIF(MSVC)

        SET(MINBUILDTIME_FLAG ON CACHE INTERNAL "Flag" FORCE)
    ENDIF()
ELSE()
    # MIN_BUILD_TIME is OFF. Change the flags back only if the flag was set before
    IF(${MINBUILDTIME_FLAG})
        MESSAGE(STATUS "MIN_BUILD_FLAG was toggled. Resetting Release FLags")
        SET(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE_DEFAULT} CACHE
            STRING "Flags used by the compiler during release builds." FORCE)
        SET(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE_DEFAULT} CACHE
            STRING "Flags used by the compiler during release builds." FORCE)
        SET(CMAKE_EXE_LINKER_FLAGS_RELEASE ${CMAKE_EXE_LINKER_FLAGS_RELEASE_DEFAULT} CACHE
            STRING "Flags used by the linker during release builds." FORCE)
        SET(CMAKE_MODULE_LINKER_FLAGS_RELEASE ${CMAKE_MODULE_LINKER_FLAGS_RELEASE_DEFAULT} CACHE
            STRING "Flags used by the linker during release builds." FORCE)
        SET(CMAKE_STATIC_LINKER_FLAGS_RELEASE ${CMAKE_STATIC_LINKER_FLAGS_RELEASE_DEFAULT} CACHE
            STRING "Flags used by the linker during release builds." FORCE)
        SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE ${CMAKE_SHARED_LINKER_FLAGS_RELEASE_DEFAULT} CACHE
            STRING "Flags used by the linker during release builds." FORCE)
        SET(MINBUILDTIME_FLAG OFF CACHE INTERNAL "Flag" FORCE)
    ENDIF()
ENDIF()

MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELEASE
    CMAKE_EXE_LINKER_FLAGS_RELEASE
    CMAKE_MODULE_LINKER_FLAGS_RELEASE
    CMAKE_STATIC_LINKER_FLAGS_RELEASE
    CMAKE_SHARED_LINKER_FLAGS_RELEASE
    )
