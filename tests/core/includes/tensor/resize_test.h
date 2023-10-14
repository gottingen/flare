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
#ifndef RESIZE_TEST_H_
#define RESIZE_TEST_H_

#include <doctest.h>
#include <flare/core.h>

namespace TestTensorResize {

    struct Default {
    };
    struct WithoutInitializing {
    };

    template<typename Tensor, typename... Args>
    inline void resize_dispatch(Default, Tensor &v, Args &&... args) {
        flare::resize(v, std::forward<Args>(args)...);
    }

    template<typename Tensor, typename... Args>
    inline void resize_dispatch(WithoutInitializing, Tensor &v, Args &&... args) {
        flare::resize(flare::WithoutInitializing, v, std::forward<Args>(args)...);
    }

    template<class DeviceType, class Tag = Default>
    void impl_testResize() {
        const size_t sizes[8] = {2, 3, 4, 5, 6, 7, 8, 9};

        // Check #904 fix (no reallocation if dimensions didn't change).
        {
            using tensor_type = flare::Tensor<int *, DeviceType>;
            tensor_type tensor_1d("tensor_1d", sizes[0]);
            const int *oldPointer = tensor_1d.data();
            REQUIRE_NE(oldPointer, nullptr);
            resize_dispatch(Tag{}, tensor_1d, sizes[0]);
            const int *newPointer = tensor_1d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int **, DeviceType>;
            tensor_type tensor_2d("tensor_2d", sizes[0], sizes[1]);
            const int *oldPointer = tensor_2d.data();
            REQUIRE_NE(oldPointer, nullptr);
            resize_dispatch(Tag{}, tensor_2d, sizes[0], sizes[1]);
            const int *newPointer = tensor_2d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int ***, DeviceType>;
            tensor_type tensor_3d("tensor_3d", sizes[0], sizes[1], sizes[2]);
            const int *oldPointer = tensor_3d.data();
            REQUIRE_NE(oldPointer, nullptr);
            resize_dispatch(Tag{}, tensor_3d, sizes[0], sizes[1], sizes[2]);
            const int *newPointer = tensor_3d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int ****, DeviceType>;
            tensor_type tensor_4d("tensor_4d", sizes[0], sizes[1], sizes[2], sizes[3]);
            const int *oldPointer = tensor_4d.data();
            REQUIRE_NE(oldPointer, nullptr);
            resize_dispatch(Tag{}, tensor_4d, sizes[0], sizes[1], sizes[2], sizes[3]);
            const int *newPointer = tensor_4d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int *****, DeviceType>;
            tensor_type tensor_5d("tensor_5d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4]);
            const int *oldPointer = tensor_5d.data();
            REQUIRE_NE(oldPointer, nullptr);
            resize_dispatch(Tag{}, tensor_5d, sizes[0], sizes[1], sizes[2], sizes[3],
                            sizes[4]);
            const int *newPointer = tensor_5d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int ******, DeviceType>;
            tensor_type tensor_6d("tensor_6d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5]);
            const int *oldPointer = tensor_6d.data();
            REQUIRE_NE(oldPointer, nullptr);
            resize_dispatch(Tag{}, tensor_6d, sizes[0], sizes[1], sizes[2], sizes[3],
                            sizes[4], sizes[5]);
            const int *newPointer = tensor_6d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int *******, DeviceType>;
            tensor_type tensor_7d("tensor_7d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5], sizes[6]);
            const int *oldPointer = tensor_7d.data();
            REQUIRE_NE(oldPointer, nullptr);
            resize_dispatch(Tag{}, tensor_7d, sizes[0], sizes[1], sizes[2], sizes[3],
                            sizes[4], sizes[5], sizes[6]);
            const int *newPointer = tensor_7d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        {
            using tensor_type = flare::Tensor<int ********, DeviceType>;
            tensor_type tensor_8d("tensor_8d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5], sizes[6], sizes[7]);
            const int *oldPointer = tensor_8d.data();
            REQUIRE_NE(oldPointer, nullptr);
            resize_dispatch(Tag{}, tensor_8d, sizes[0], sizes[1], sizes[2], sizes[3],
                            sizes[4], sizes[5], sizes[6], sizes[7]);
            const int *newPointer = tensor_8d.data();
            REQUIRE_EQ(oldPointer, newPointer);
        }
        // Resize without initialization: check if data preserved
        {
            using tensor_type = flare::Tensor<int *, DeviceType>;
            tensor_type tensor_1d("tensor_1d", sizes[0]);
            typename tensor_type::HostMirror h_tensor_1d_old =
                    flare::create_mirror(tensor_1d);
            flare::deep_copy(tensor_1d, 111);
            flare::deep_copy(h_tensor_1d_old, tensor_1d);
            resize_dispatch(Tag{}, tensor_1d, 2 * sizes[0]);
            REQUIRE_EQ(tensor_1d.extent(0), 2 * sizes[0]);
            typename tensor_type::HostMirror h_tensor_1d =
                    flare::create_mirror_tensor(tensor_1d);
            flare::deep_copy(h_tensor_1d, tensor_1d);
            bool test = true;
            for (size_t i0 = 0; i0 < sizes[0]; ++i0) {
                if (h_tensor_1d(i0) != h_tensor_1d_old(i0)) {
                    test = false;
                    break;
                }
            }
            REQUIRE(test);
        }
        {
            using tensor_type = flare::Tensor<int **, DeviceType>;
            tensor_type tensor_2d("tensor_2d", sizes[0], sizes[1]);
            typename tensor_type::HostMirror h_tensor_2d_old =
                    flare::create_mirror(tensor_2d);
            flare::deep_copy(tensor_2d, 222);
            flare::deep_copy(h_tensor_2d_old, tensor_2d);
            resize_dispatch(Tag{}, tensor_2d, 2 * sizes[0], sizes[1]);
            REQUIRE_EQ(tensor_2d.extent(0), 2 * sizes[0]);
            typename tensor_type::HostMirror h_tensor_2d =
                    flare::create_mirror_tensor(tensor_2d);
            flare::deep_copy(h_tensor_2d, tensor_2d);
            bool test = true;
            for (size_t i0 = 0; i0 < sizes[0]; ++i0) {
                for (size_t i1 = 0; i1 < sizes[1]; ++i1) {
                    if (h_tensor_2d(i0, i1) != h_tensor_2d_old(i0, i1)) {
                        test = false;
                        break;
                    }
                }
            }
            REQUIRE(test);
        }
        {
            using tensor_type = flare::Tensor<int ***, DeviceType>;
            tensor_type tensor_3d("tensor_3d", sizes[0], sizes[1], sizes[2]);
            typename tensor_type::HostMirror h_tensor_3d_old =
                    flare::create_mirror(tensor_3d);
            flare::deep_copy(tensor_3d, 333);
            flare::deep_copy(h_tensor_3d_old, tensor_3d);
            resize_dispatch(Tag{}, tensor_3d, 2 * sizes[0], sizes[1], sizes[2]);
            REQUIRE_EQ(tensor_3d.extent(0), 2 * sizes[0]);
            typename tensor_type::HostMirror h_tensor_3d =
                    flare::create_mirror_tensor(tensor_3d);
            flare::deep_copy(h_tensor_3d, tensor_3d);
            bool test = true;
            for (size_t i0 = 0; i0 < sizes[0]; ++i0) {
                for (size_t i1 = 0; i1 < sizes[1]; ++i1) {
                    for (size_t i2 = 0; i2 < sizes[2]; ++i2) {
                        if (h_tensor_3d(i0, i1, i2) != h_tensor_3d_old(i0, i1, i2)) {
                            test = false;
                            break;
                        }
                    }
                }
            }
            REQUIRE(test);
        }
        {
            using tensor_type = flare::Tensor<int ****, DeviceType>;
            tensor_type tensor_4d("tensor_4d", sizes[0], sizes[1], sizes[2], sizes[3]);
            typename tensor_type::HostMirror h_tensor_4d_old =
                    flare::create_mirror(tensor_4d);
            flare::deep_copy(tensor_4d, 444);
            flare::deep_copy(h_tensor_4d_old, tensor_4d);
            resize_dispatch(Tag{}, tensor_4d, 2 * sizes[0], sizes[1], sizes[2], sizes[3]);
            REQUIRE_EQ(tensor_4d.extent(0), 2 * sizes[0]);
            typename tensor_type::HostMirror h_tensor_4d =
                    flare::create_mirror_tensor(tensor_4d);
            flare::deep_copy(h_tensor_4d, tensor_4d);
            bool test = true;
            for (size_t i0 = 0; i0 < sizes[0]; ++i0) {
                for (size_t i1 = 0; i1 < sizes[1]; ++i1) {
                    for (size_t i2 = 0; i2 < sizes[2]; ++i2) {
                        for (size_t i3 = 0; i3 < sizes[3]; ++i3) {
                            if (h_tensor_4d(i0, i1, i2, i3) != h_tensor_4d_old(i0, i1, i2, i3)) {
                                test = false;
                                break;
                            }
                        }
                    }
                }
            }
            REQUIRE(test);
        }
        {
            using tensor_type = flare::Tensor<int *****, DeviceType>;
            tensor_type tensor_5d("tensor_5d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4]);
            typename tensor_type::HostMirror h_tensor_5d_old =
                    flare::create_mirror(tensor_5d);
            flare::deep_copy(tensor_5d, 555);
            flare::deep_copy(h_tensor_5d_old, tensor_5d);
            resize_dispatch(Tag{}, tensor_5d, 2 * sizes[0], sizes[1], sizes[2], sizes[3],
                            sizes[4]);
            REQUIRE_EQ(tensor_5d.extent(0), 2 * sizes[0]);
            typename tensor_type::HostMirror h_tensor_5d =
                    flare::create_mirror_tensor(tensor_5d);
            flare::deep_copy(h_tensor_5d, tensor_5d);
            bool test = true;
            for (size_t i0 = 0; i0 < sizes[0]; ++i0) {
                for (size_t i1 = 0; i1 < sizes[1]; ++i1) {
                    for (size_t i2 = 0; i2 < sizes[2]; ++i2) {
                        for (size_t i3 = 0; i3 < sizes[3]; ++i3) {
                            for (size_t i4 = 0; i4 < sizes[4]; ++i4) {
                                if (h_tensor_5d(i0, i1, i2, i3, i4) !=
                                    h_tensor_5d_old(i0, i1, i2, i3, i4)) {
                                    test = false;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            REQUIRE(test);
        }
        {
            using tensor_type = flare::Tensor<int ******, DeviceType>;
            tensor_type tensor_6d("tensor_6d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5]);
            typename tensor_type::HostMirror h_tensor_6d_old =
                    flare::create_mirror(tensor_6d);
            flare::deep_copy(tensor_6d, 666);
            flare::deep_copy(h_tensor_6d_old, tensor_6d);
            resize_dispatch(Tag{}, tensor_6d, 2 * sizes[0], sizes[1], sizes[2], sizes[3],
                            sizes[4], sizes[5]);
            REQUIRE_EQ(tensor_6d.extent(0), 2 * sizes[0]);
            typename tensor_type::HostMirror h_tensor_6d =
                    flare::create_mirror_tensor(tensor_6d);
            flare::deep_copy(h_tensor_6d, tensor_6d);
            bool test = true;
            for (size_t i0 = 0; i0 < sizes[0]; ++i0) {
                for (size_t i1 = 0; i1 < sizes[1]; ++i1) {
                    for (size_t i2 = 0; i2 < sizes[2]; ++i2) {
                        for (size_t i3 = 0; i3 < sizes[3]; ++i3) {
                            for (size_t i4 = 0; i4 < sizes[4]; ++i4) {
                                for (size_t i5 = 0; i5 < sizes[5]; ++i5) {
                                    if (h_tensor_6d(i0, i1, i2, i3, i4, i5) !=
                                        h_tensor_6d_old(i0, i1, i2, i3, i4, i5)) {
                                        test = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            REQUIRE(test);
        }
        {
            using tensor_type = flare::Tensor<int *******, DeviceType>;
            tensor_type tensor_7d("tensor_7d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5], sizes[6]);
            typename tensor_type::HostMirror h_tensor_7d_old =
                    flare::create_mirror(tensor_7d);
            flare::deep_copy(tensor_7d, 777);
            flare::deep_copy(h_tensor_7d_old, tensor_7d);
            resize_dispatch(Tag{}, tensor_7d, 2 * sizes[0], sizes[1], sizes[2], sizes[3],
                            sizes[4], sizes[5], sizes[6]);
            REQUIRE_EQ(tensor_7d.extent(0), 2 * sizes[0]);
            typename tensor_type::HostMirror h_tensor_7d =
                    flare::create_mirror_tensor(tensor_7d);
            flare::deep_copy(h_tensor_7d, tensor_7d);
            bool test = true;
            for (size_t i0 = 0; i0 < sizes[0]; ++i0) {
                for (size_t i1 = 0; i1 < sizes[1]; ++i1) {
                    for (size_t i2 = 0; i2 < sizes[2]; ++i2) {
                        for (size_t i3 = 0; i3 < sizes[3]; ++i3) {
                            for (size_t i4 = 0; i4 < sizes[4]; ++i4) {
                                for (size_t i5 = 0; i5 < sizes[5]; ++i5) {
                                    for (size_t i6 = 0; i6 < sizes[6]; ++i6) {
                                        if (h_tensor_7d(i0, i1, i2, i3, i4, i5, i6) !=
                                            h_tensor_7d_old(i0, i1, i2, i3, i4, i5, i6)) {
                                            test = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            REQUIRE(test);
        }
        {
            using tensor_type = flare::Tensor<int ********, DeviceType>;
            tensor_type tensor_8d("tensor_8d", sizes[0], sizes[1], sizes[2], sizes[3],
                              sizes[4], sizes[5], sizes[6], sizes[7]);
            typename tensor_type::HostMirror h_tensor_8d_old =
                    flare::create_mirror(tensor_8d);
            flare::deep_copy(tensor_8d, 888);
            flare::deep_copy(h_tensor_8d_old, tensor_8d);
            resize_dispatch(Tag{}, tensor_8d, 2 * sizes[0], sizes[1], sizes[2], sizes[3],
                            sizes[4], sizes[5], sizes[6], sizes[7]);
            REQUIRE_EQ(tensor_8d.extent(0), 2 * sizes[0]);
            typename tensor_type::HostMirror h_tensor_8d =
                    flare::create_mirror_tensor(tensor_8d);
            flare::deep_copy(h_tensor_8d, tensor_8d);
            bool test = true;
            for (size_t i0 = 0; i0 < sizes[0]; ++i0) {
                for (size_t i1 = 0; i1 < sizes[1]; ++i1) {
                    for (size_t i2 = 0; i2 < sizes[2]; ++i2) {
                        for (size_t i3 = 0; i3 < sizes[3]; ++i3) {
                            for (size_t i4 = 0; i4 < sizes[4]; ++i4) {
                                for (size_t i5 = 0; i5 < sizes[5]; ++i5) {
                                    for (size_t i6 = 0; i6 < sizes[6]; ++i6) {
                                        for (size_t i7 = 0; i7 < sizes[7]; ++i7) {
                                            if (h_tensor_8d(i0, i1, i2, i3, i4, i5, i6, i7) !=
                                                h_tensor_8d_old(i0, i1, i2, i3, i4, i5, i6, i7)) {
                                                test = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            REQUIRE(test);
        }
    }

    template<class DeviceType>
    void testResize() {
        {
            impl_testResize<DeviceType>();  // with data initialization
        }
        {
            impl_testResize<DeviceType,
                    WithoutInitializing>();  // without data initialization
        }
    }

}  // namespace TestTensorResize
#endif  // RESIZE_TEST_H_
