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

#pragma once

namespace flare {
namespace common {
template<typename T, typename H>
class HandleBase {
    H handle_;

   public:
    HandleBase() : handle_(0) { static_cast<T*>(this)->createHandle(&handle_); }
    ~HandleBase() { static_cast<T*>(this)->destroyHandle(handle_); }

    operator H() { return handle_; }
    H* get() { return &handle_; }

    HandleBase(HandleBase const&)     = delete;
    void operator=(HandleBase const&) = delete;

    HandleBase(HandleBase&& h)            = default;
    HandleBase& operator=(HandleBase&& h) = default;
};
}  // namespace common
}  // namespace flare

#define CREATE_HANDLE(NAME, TYPE, CREATE_FUNCTION, DESTROY_FUNCTION,  \
                      CHECK_FUNCTION)                                 \
    class NAME : public common::HandleBase<NAME, TYPE> {              \
       public:                                                        \
        void createHandle(TYPE* handle) {                             \
            CHECK_FUNCTION(CREATE_FUNCTION(handle));                  \
        }                                                             \
        void destroyHandle(TYPE handle) { DESTROY_FUNCTION(handle); } \
    };
