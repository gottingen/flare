// Copyright 2023 The Elastic-AI Authors.
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


// This file is needed in order to get the linker language
// for the header only submodule.
// While we set the language properties in our normal cmake
// path it does not get set in the Trilinos environment.
// Furthermore, setting LINKER_LANGUAGE is only supported
// in CMAKE 3.19 and up.
void FLARE_SIMD_SRC_DUMMY_PREVENT_LINK_ERROR() {}
