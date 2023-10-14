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
#ifndef REALLOC_TEST_H_
#define REALLOC_TEST_H_

#include <doctest.h>
#include <flare/core.h>

namespace TestTensorRealloc {

    struct Default {
    };
    struct WithoutInitializing {
    };

    template<typename Tensor, typename... Args>
    inline void realloc_dispatch(Default, Tensor &v, Args &&... args) {
        flare::realloc(v, std::forward<Args>(args)...);
    }

    template<typename Tensor, typename... Args>
    inline void realloc_dispatch(WithoutInitializing, Tensor &v, Args &&... args) {
        flare::realloc(flare::WithoutInitializing, v, std::forward<Args>(args)...);
    }

    template<class DeviceType, class Tag = Default>
    void impl_testRealloc() {
        const size_t sizes[8] = {2, 3, 4, 5, 6, 7, 8, 9};

        // Check #904 fix (no reallocation if dimensions didn't change).
        {
            using tensor_type = flare::Tensor<int *, DeviceType>;
            tensor_type tensor_1d("tensor_1d", sizes[0]);
            const int *oldPointer = tensor_1d.data();
            auto const &oldLabel = tensor_1d.label();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, tensor_1d, sizes[0]);
            auto const &newLabel = tensor_1d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = tensor_1d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int **, DeviceType>;
            tensor_type tensor_2d("tensor_2d", sizes[0], sizes[1]);
            auto const &oldLabel = tensor_2d.label();
            const int *oldPointer = tensor_2d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, tensor_2d, sizes[0], sizes[1]);
            auto const &newLabel = tensor_2d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = tensor_2d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int ***, DeviceType>;
            tensor_type tensor_3d("tensor_3d", sizes[0], sizes[1], sizes[2]);
            auto const &oldLabel = tensor_3d.label();
            const int *oldPointer = tensor_3d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, tensor_3d, sizes[0], sizes[1], sizes[2]);
            auto const &newLabel = tensor_3d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = tensor_3d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int ****, DeviceType>;
            tensor_type tensor_4d("tensor_4d", sizes[0], sizes[1], sizes[2], sizes[3]);
            auto const &oldLabel = tensor_4d.label();
            const int *oldPointer = tensor_4d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, tensor_4d, sizes[0], sizes[1], sizes[2], sizes[3]);
            auto const &newLabel = tensor_4d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = tensor_4d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int *****, DeviceType>;
            tensor_type tensor_5d("tensor_5d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4]);
            auto const &oldLabel = tensor_5d.label();
            const int *oldPointer = tensor_5d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, tensor_5d, sizes[0], sizes[1], sizes[2], sizes[3],
                             sizes[4]);
            auto const &newLabel = tensor_5d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = tensor_5d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int ******, DeviceType>;
            tensor_type tensor_6d("tensor_6d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5]);
            const int *oldPointer = tensor_6d.data();
            auto const &oldLabel = tensor_6d.label();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, tensor_6d, sizes[0], sizes[1], sizes[2], sizes[3],
                             sizes[4], sizes[5]);
            auto const &newLabel = tensor_6d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = tensor_6d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int *******, DeviceType>;
            tensor_type tensor_7d("tensor_7d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5], sizes[6]);
            auto const &oldLabel = tensor_7d.label();
            const int *oldPointer = tensor_7d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, tensor_7d, sizes[0], sizes[1], sizes[2], sizes[3],
                             sizes[4], sizes[5], sizes[6]);
            auto const &newLabel = tensor_7d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = tensor_7d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int ********, DeviceType>;
            tensor_type tensor_8d("tensor_8d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5], sizes[6], sizes[7]);
            auto const &oldLabel = tensor_8d.label();
            const int *oldPointer = tensor_8d.data();
            REQUIRE_NE(oldPointer, nullptr);
            realloc_dispatch(Tag{}, tensor_8d, sizes[0], sizes[1], sizes[2], sizes[3],
                             sizes[4], sizes[5], sizes[6], sizes[7]);
            auto const &newLabel = tensor_8d.label();
            REQUIRE_EQ(oldLabel, newLabel);
            const int *newPointer = tensor_8d.data();
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

}  // namespace TestTensorRealloc
#endif  // REALLOC_TEST_H_
