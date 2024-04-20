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
# CMake 3.9 or later provides a global property to whether we are multi-config
# or single-config generator. Before 3.9, the defintion of CMAKE_CONFIGURATION_TYPES
# variable indicated multi-config, but developers might modify.
if(NOT CMAKE_VERSION VERSION_LESS 3.9)
  get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
elseif(CMAKE_CONFIGURATION_TYPES)
  # CMAKE_CONFIGURATION_TYPES is set by project() call for multi-config generators
  set(isMultiConfig True)
else()
  set(isMultiConfig False)
endif()

if(isMultiConfig)
  set(CMAKE_CONFIGURATION_TYPES
    "Coverage;Debug;MinSizeRel;Release;RelWithDebInfo"
    CACHE STRING "Configurations for Multi-Config CMake Generator" FORCE)
else()
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE
    PROPERTY
      STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Coverage")
endif()
