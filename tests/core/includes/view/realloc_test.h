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
#ifndef TESTREALLOC_HPP_
#define TESTREALLOC_HPP_

#include <doctest.h>
#include <flare/core.h>

namespace TestViewRealloc {

    struct Default {
    };
    struct WithoutInitializing {
    };

    template<typename View, typename... Args>
    inline void realloc_dispatch(Default, View &v, Args &&... args) {
        flare::realloc(v, std::forward<Args>(args)...);
    }

    template<typename View, typename... Args>
    inline void realloc_dispatch(WithoutInitializing, View &v, Args &&... args) {
        flare::realloc(flare::WithoutInitializing, v, std::forward<Args>(args)...);
    }

    template<class DeviceType, class Tag = Default>
    void impl_testRealloc() {
        const size_t sizes[8] = {2, 3, 4, 5, 6, 7, 8, 9};

        // Check #904 fix (no reallocation if dimensions didn't change).
        {
            using view_type = flare::View<int *, DeviceType>;
            view_type view_1d("view_1d", sizes[0]);
            const int *oldPointer = view_1d.data();
            auto const &oldLabel = view_1d.label();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, view_1d, sizes[0]);
            auto const &newLabel = view_1d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = view_1d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using view_type = flare::View<int **, DeviceType>;
            view_type view_2d("view_2d", sizes[0], sizes[1]);
            auto const &oldLabel = view_2d.label();
            const int *oldPointer = view_2d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, view_2d, sizes[0], sizes[1]);
            auto const &newLabel = view_2d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = view_2d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using view_type = flare::View<int ***, DeviceType>;
            view_type view_3d("view_3d", sizes[0], sizes[1], sizes[2]);
            auto const &oldLabel = view_3d.label();
            const int *oldPointer = view_3d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, view_3d, sizes[0], sizes[1], sizes[2]);
            auto const &newLabel = view_3d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = view_3d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using view_type = flare::View<int ****, DeviceType>;
            view_type view_4d("view_4d", sizes[0], sizes[1], sizes[2], sizes[3]);
            auto const &oldLabel = view_4d.label();
            const int *oldPointer = view_4d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, view_4d, sizes[0], sizes[1], sizes[2], sizes[3]);
            auto const &newLabel = view_4d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = view_4d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using view_type = flare::View<int *****, DeviceType>;
            view_type view_5d("view_5d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4]);
            auto const &oldLabel = view_5d.label();
            const int *oldPointer = view_5d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, view_5d, sizes[0], sizes[1], sizes[2], sizes[3],
                             sizes[4]);
            auto const &newLabel = view_5d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = view_5d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using view_type = flare::View<int ******, DeviceType>;
            view_type view_6d("view_6d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5]);
            const int *oldPointer = view_6d.data();
            auto const &oldLabel = view_6d.label();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, view_6d, sizes[0], sizes[1], sizes[2], sizes[3],
                             sizes[4], sizes[5]);
            auto const &newLabel = view_6d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = view_6d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using view_type = flare::View<int *******, DeviceType>;
            view_type view_7d("view_7d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5], sizes[6]);
            auto const &oldLabel = view_7d.label();
            const int *oldPointer = view_7d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, view_7d, sizes[0], sizes[1], sizes[2], sizes[3],
                             sizes[4], sizes[5], sizes[6]);
            auto const &newLabel = view_7d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = view_7d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using view_type = flare::View<int ********, DeviceType>;
            view_type view_8d("view_8d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5], sizes[6], sizes[7]);
            auto const &oldLabel = view_8d.label();
            const int *oldPointer = view_8d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, view_8d, sizes[0], sizes[1], sizes[2], sizes[3],
                             sizes[4], sizes[5], sizes[6], sizes[7]);
            auto const &newLabel = view_8d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = view_8d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
    }

    template<class DeviceType>
    void testRealloc() {
        {
            impl_testRealloc<DeviceType>();  // with data initialization
        }
        {
            impl_testRealloc<DeviceType,
                    WithoutInitializing>();  // without data initialization
        }
    }

}  // namespace TestViewRealloc
#endif  // TESTREALLOC_HPP_
