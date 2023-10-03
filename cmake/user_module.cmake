#
# Copyright 2023 The titan-search Authors.
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

include(user_deps)
include(user_cxx_config)

set(FLARE_BUILD_DEVICES)
if (FLARE_BUILD_CUDA)
    list(APPEND FLARE_BUILD_DEVICES cuda)
endif ()

if (FLARE_BUILD_OPENMP)
    list(APPEND FLARE_BUILD_DEVICES openmp)
endif ()

if (FLARE_BUILD_THREADS)
    list(APPEND FLARE_BUILD_DEVICES threads)
endif ()

if (FLARE_BUILD_SERIAL)
    list(APPEND FLARE_BUILD_DEVICES serial)
endif ()

carbin_print_list_label("FLARE_BUILD_DEVICES" FLARE_BUILD_DEVICES)