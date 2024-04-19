// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <build_version.hpp>
#include <fly/util.h>

fly_err fly_get_version(int *major, int *minor, int *patch) {
    *major = FLY_VERSION_MAJOR;
    *minor = FLY_VERSION_MINOR;
    *patch = FLY_VERSION_PATCH;

    return FLY_SUCCESS;
}

const char *fly_get_revision() { return FLY_REVISION; }
