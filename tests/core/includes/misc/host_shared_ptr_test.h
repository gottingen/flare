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

#include <flare/core/memory/host_shared_ptr.h>

#include <doctest.h>

using flare::detail::HostSharedPtr;

TEST_CASE("TEST_CATEGORY, host_shared_ptr_use_count") {
    using T = int;
    {
        HostSharedPtr<T> p1;
        REQUIRE_EQ(p1.use_count(), 0);
    }
    {
        HostSharedPtr<T> p1(nullptr);
        REQUIRE_EQ(p1.use_count(), 0);
    }
    {
        HostSharedPtr<T> p1(new T());
        REQUIRE_EQ(p1.use_count(), 1);
    }
    {
        HostSharedPtr<T> p1(new T(), [](T *p) { delete p; });
        REQUIRE_EQ(p1.use_count(), 1);
    }
    {
        T i;
        HostSharedPtr<T> p1(&i, [](T *) {});
        REQUIRE_EQ(p1.use_count(), 1);
    }
    {
        HostSharedPtr<T> p1(new T());
        HostSharedPtr<T> p2(p1);  // copy construction
        REQUIRE_EQ(p1.use_count(), 2);
        REQUIRE_EQ(p2.use_count(), 2);
    }
    {
        HostSharedPtr<T> p1(new T());
        HostSharedPtr<T> p2(std::move(p1));  // move construction
        REQUIRE_EQ(p2.use_count(), 1);
    }
    {
        HostSharedPtr<T> p1(new T());
        HostSharedPtr<T> p2;
        p2 = p1;  // copy assignment
        REQUIRE_EQ(p1.use_count(), 2);
        REQUIRE_EQ(p2.use_count(), 2);
    }
    {
        HostSharedPtr<T> p1(new T());
        HostSharedPtr<T> p2;
        p2 = std::move(p1);  // move assignment
        REQUIRE_EQ(p2.use_count(), 1);
    }
}

TEST_CASE("TEST_CATEGORY, host_shared_ptr_get") {
    using T = int;
    {
        HostSharedPtr<T> p1;
        REQUIRE_EQ(p1.get(), nullptr);
    }
    {
        HostSharedPtr<T> p1(nullptr);
        REQUIRE_EQ(p1.get(), nullptr);
    }
    {
        T *p_i = new T();
        HostSharedPtr<T> p1(p_i);
        REQUIRE_EQ(p1.get(), p_i);
    }
    {
        T *p_i = new T();
        HostSharedPtr<T> p1(p_i, [](T *p) { delete p; });
        REQUIRE_EQ(p1.get(), p_i);
    }
    {
        T i;
        HostSharedPtr<T> p1(&i, [](T *) {});
        REQUIRE_EQ(p1.get(), &i);
    }
    {
        T i;
        HostSharedPtr<T> p1(&i, [](T *) {});
        HostSharedPtr<T> p2(p1);  // copy construction
        REQUIRE_EQ(p1.get(), &i);
        REQUIRE_EQ(p1.get(), &i);
    }
    {
        T i;
        HostSharedPtr<T> p1(&i, [](T *) {});
        HostSharedPtr<T> p2(std::move(p1));  // move construction
        REQUIRE_EQ(p1.get(), nullptr);
        REQUIRE_EQ(p2.get(), &i);
    }
    {
        T i;
        HostSharedPtr<T> p1(&i, [](T *) {});
        HostSharedPtr<T> p2;
        p2 = p1;  // copy assignment
        REQUIRE_EQ(p1.get(), &i);
        REQUIRE_EQ(p2.get(), &i);
    }
    {
        T i;
        HostSharedPtr<T> p1(&i, [](T *) {});
        HostSharedPtr<T> p2;
        p2 = std::move(p1);  // move assignment
        REQUIRE_EQ(p1.get(), nullptr);
        REQUIRE_EQ(p2.get(), &i);
    }
}
