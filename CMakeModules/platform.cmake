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


# Platform specific settings
#
# Add paths and flags specific platforms. This can inc

if(APPLE)
  # Default path for Intel MKL libraries
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/intel/mkl/lib")
endif()

if(UNIX AND NOT APPLE)
  # Default path for Intel MKL libraries
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/intel/oneapi/mkl/latest/lib")
endif()

