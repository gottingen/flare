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

#include <flare/core.h>
#include <cuda_category_test.h>
#include <doctest.h>

namespace Test {

    __global__ void test_abort() { flare::abort("test_abort"); }

    __global__ void test_cuda_spaces_int_value(int *ptr) {
        if (*ptr == 42) {
            *ptr = 2 * 42;
        }
    }

    TEST_CASE("cuda, space_access") {
        static_assert(flare::detail::MemorySpaceAccess<flare::HostSpace,
                              flare::HostSpace>::assignable,
                      "");

        static_assert(
                flare::detail::MemorySpaceAccess<flare::HostSpace,
                        flare::CudaHostPinnedSpace>::assignable,
                "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::HostSpace,
                              flare::CudaSpace>::assignable,
                      "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::HostSpace,
                              flare::CudaSpace>::accessible,
                      "");

        static_assert(
                !flare::detail::MemorySpaceAccess<flare::HostSpace,
                        flare::CudaUVMSpace>::assignable,
                "");

        static_assert(
                flare::detail::MemorySpaceAccess<flare::HostSpace,
                        flare::CudaUVMSpace>::accessible,
                "");

        //--------------------------------------

        static_assert(flare::detail::MemorySpaceAccess<flare::CudaSpace,
                              flare::CudaSpace>::assignable,
                      "");

        static_assert(
                flare::detail::MemorySpaceAccess<flare::CudaSpace,
                        flare::CudaUVMSpace>::assignable,
                "");

        static_assert(
                !flare::detail::MemorySpaceAccess<flare::CudaSpace,
                        flare::CudaHostPinnedSpace>::assignable,
                "");

        static_assert(
                flare::detail::MemorySpaceAccess<flare::CudaSpace,
                        flare::CudaHostPinnedSpace>::accessible,
                "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::CudaSpace,
                              flare::HostSpace>::assignable,
                      "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::CudaSpace,
                              flare::HostSpace>::accessible,
                      "");

        //--------------------------------------

        static_assert(
                flare::detail::MemorySpaceAccess<flare::CudaUVMSpace,
                        flare::CudaUVMSpace>::assignable,
                "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::CudaUVMSpace,
                              flare::CudaSpace>::assignable,
                      "");

        static_assert(flare::detail::MemorySpaceAccess<flare::CudaUVMSpace,
                              flare::CudaSpace>::accessible,
                      "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::CudaUVMSpace,
                              flare::HostSpace>::assignable,
                      "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::CudaUVMSpace,
                              flare::HostSpace>::accessible,
                      "");

        static_assert(
                !flare::detail::MemorySpaceAccess<flare::CudaUVMSpace,
                        flare::CudaHostPinnedSpace>::assignable,
                "");

        static_assert(
                flare::detail::MemorySpaceAccess<flare::CudaUVMSpace,
                        flare::CudaHostPinnedSpace>::accessible,
                "");

        //--------------------------------------

        static_assert(
                flare::detail::MemorySpaceAccess<flare::CudaHostPinnedSpace,
                        flare::CudaHostPinnedSpace>::assignable,
                "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::CudaHostPinnedSpace,
                              flare::HostSpace>::assignable,
                      "");

        static_assert(flare::detail::MemorySpaceAccess<flare::CudaHostPinnedSpace,
                              flare::HostSpace>::accessible,
                      "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::CudaHostPinnedSpace,
                              flare::CudaSpace>::assignable,
                      "");

        static_assert(!flare::detail::MemorySpaceAccess<flare::CudaHostPinnedSpace,
                              flare::CudaSpace>::accessible,
                      "");

        static_assert(
                !flare::detail::MemorySpaceAccess<flare::CudaHostPinnedSpace,
                        flare::CudaUVMSpace>::assignable,
                "");

        static_assert(
                flare::detail::MemorySpaceAccess<flare::CudaHostPinnedSpace,
                        flare::CudaUVMSpace>::accessible,
                "");

        //--------------------------------------

        static_assert(
                !flare::SpaceAccessibility<flare::Cuda, flare::HostSpace>::accessible,
                "");

        static_assert(
                flare::SpaceAccessibility<flare::Cuda, flare::CudaSpace>::accessible,
                "");

        static_assert(flare::SpaceAccessibility<flare::Cuda,
                              flare::CudaUVMSpace>::accessible,
                      "");

        static_assert(
                flare::SpaceAccessibility<flare::Cuda,
                        flare::CudaHostPinnedSpace>::accessible,
                "");

        static_assert(!flare::SpaceAccessibility<flare::HostSpace,
                              flare::CudaSpace>::accessible,
                      "");

        static_assert(flare::SpaceAccessibility<flare::HostSpace,
                              flare::CudaUVMSpace>::accessible,
                      "");

        static_assert(
                flare::SpaceAccessibility<flare::HostSpace,
                        flare::CudaHostPinnedSpace>::accessible,
                "");

        static_assert(std::is_same<flare::detail::HostMirror<flare::CudaSpace>::Space,
                              flare::HostSpace>::value,
                      "");

        static_assert(
                std::is_same<flare::detail::HostMirror<flare::CudaUVMSpace>::Space,
                        flare::Device<flare::HostSpace::execution_space,
                                flare::CudaUVMSpace>>::value,
                "");

        static_assert(
                std::is_same<flare::detail::HostMirror<flare::CudaHostPinnedSpace>::Space,
                        flare::CudaHostPinnedSpace>::value,
                "");

        static_assert(std::is_same<flare::Device<flare::HostSpace::execution_space,
                              flare::CudaUVMSpace>,
                              flare::Device<flare::HostSpace::execution_space,
                                      flare::CudaUVMSpace>>::value,
                      "");

        static_assert(
                flare::SpaceAccessibility<flare::detail::HostMirror<flare::Cuda>::Space,
                        flare::HostSpace>::accessible,
                "");

        static_assert(flare::SpaceAccessibility<
                              flare::detail::HostMirror<flare::CudaSpace>::Space,
                              flare::HostSpace>::accessible,
                      "");

        static_assert(flare::SpaceAccessibility<
                              flare::detail::HostMirror<flare::CudaUVMSpace>::Space,
                              flare::HostSpace>::accessible,
                      "");

        static_assert(
                flare::SpaceAccessibility<
                        flare::detail::HostMirror<flare::CudaHostPinnedSpace>::Space,
                        flare::HostSpace>::accessible,
                "");
    }

    TEST_CASE("cuda, uvm") {
        int *uvm_ptr = static_cast<int *>(
                flare::flare_malloc<flare::CudaUVMSpace>("uvm_ptr", sizeof(int)));

        *uvm_ptr = 42;

        flare::fence();
        test_cuda_spaces_int_value<<<1, 1>>>(uvm_ptr);
        flare::fence();

        REQUIRE_EQ(*uvm_ptr, int(2 * 42));

        flare::flare_free<flare::CudaUVMSpace>(uvm_ptr);
    }

    template<class MemSpace, class ExecSpace>
    struct TestViewCudaAccessible {
        enum {
            N = 1000
        };

        using V = flare::View<double *, MemSpace>;

        V m_base;

        struct TagInit {
        };
        struct TagTest {
        };

        FLARE_INLINE_FUNCTION
        void operator()(const TagInit &, const int i) const { m_base[i] = i + 1; }

        FLARE_INLINE_FUNCTION
        void operator()(const TagTest &, const int i, long &error_count) const {
            if (m_base[i] != i + 1) ++error_count;
        }

        TestViewCudaAccessible() : m_base("base", N) {}

        static void run() {
            TestViewCudaAccessible self;
            flare::parallel_for(
                    flare::RangePolicy<typename MemSpace::execution_space, TagInit>(0, N),
                    self);
            typename MemSpace::execution_space().fence();

            // Next access is a different execution space, must complete prior kernel.
            long error_count = -1;
            flare::parallel_reduce(flare::RangePolicy<ExecSpace, TagTest>(0, N), self,
                                   error_count);
            REQUIRE_EQ(error_count, 0);
        }
    };

    TEST_CASE("cuda, impl_view_accessible") {
        TestViewCudaAccessible<flare::CudaSpace, flare::Cuda>::run();

        TestViewCudaAccessible<flare::CudaUVMSpace, flare::Cuda>::run();
        TestViewCudaAccessible<flare::CudaUVMSpace,
                flare::HostSpace::execution_space>::run();

        TestViewCudaAccessible<flare::CudaHostPinnedSpace, flare::Cuda>::run();
        TestViewCudaAccessible<flare::CudaHostPinnedSpace,
                flare::HostSpace::execution_space>::run();
    }

    template<class MemSpace>
    struct TestViewCudaTexture {
        enum {
            N = 1000
        };

        using V = flare::View<double *, MemSpace>;
        using T = flare::View<const double *, MemSpace, flare::MemoryRandomAccess>;

        V m_base;
        T m_tex;

        struct TagInit {
        };
        struct TagTest {
        };

        FLARE_INLINE_FUNCTION
        void operator()(const TagInit &, const int i) const { m_base[i] = i + 1; }

        FLARE_INLINE_FUNCTION
        void operator()(const TagTest &, const int i, long &error_count) const {
            if (m_tex[i] != i + 1) ++error_count;
        }

        TestViewCudaTexture() : m_base("base", N), m_tex(m_base) {}

        static void run() {
            REQUIRE((std::is_same<typename V::reference_type, double &>::value));
            REQUIRE(
                    (std::is_same<typename T::reference_type, const double>::value));

            REQUIRE(V::reference_type_is_lvalue_reference);   // An ordinary view.
            REQUIRE_FALSE(T::reference_type_is_lvalue_reference);  // Texture fetch
            // returns by value.

            TestViewCudaTexture self;
            flare::parallel_for(flare::RangePolicy<flare::Cuda, TagInit>(0, N),
                                self);

            long error_count = -1;
            flare::parallel_reduce(flare::RangePolicy<flare::Cuda, TagTest>(0, N),
                                   self, error_count);
            REQUIRE_EQ(error_count, 0);
        }
    };

    TEST_CASE("cuda, impl_view_texture") {
        TestViewCudaTexture<flare::CudaSpace>::run();
        TestViewCudaTexture<flare::CudaUVMSpace>::run();
    }

    // couldn't create a random-access subview of a view of const T in flare::Cuda
    namespace issue_5594 {

        template<typename View>
        struct InitFunctor {
            InitFunctor(const View &view) : view_(view) {}

            FLARE_INLINE_FUNCTION
            void operator()(int i) const { view_(i) = i; }

            View view_;
        };

        template<typename V1, typename V2>
        struct Issue5594Functor {
            Issue5594Functor(const V1 &v1) : v1_(v1) {}

            FLARE_INLINE_FUNCTION
            void operator()(int i, int &lerr) const {
                V2 v2(&v1_(0),
                      v1_.size());  // failure here -- create subview in execution space
                lerr += v1_(i) != v2(i);  // check that subview is correct
            }

            V1 v1_;
        };

        template<typename View>
        View create_view() {
            using execution_space = typename View::execution_space;
            View view("", 10);
            // MSVC+CUDA errors on CTAD here
            InitFunctor<View> iota(view);
            flare::parallel_for("test_view_subview_const_randomaccess",
                                flare::RangePolicy<execution_space>(0, view.extent(0)),
                                iota);
            return view;
        }

// creating a RandomAccess subview of a view of const T in flare::Cuda
        template<typename Exec, typename Mem>
        void test_view_subview_const_randomaccess() {
            using view_t = flare::View<int *, Mem>;
            using view_const_t = flare::View<const int *, Mem>;
            using u_view_const_t = flare::View<
                    const int *, Mem,
                    flare::MemoryTraits<flare::Unmanaged | flare::RandomAccess>>;

            // create non-const view with known values
            view_t nonConst = create_view<view_t>();
            // get a const version of the values
            view_const_t view(nonConst);

            // create a subview in the execution space and check that it worked
            Issue5594Functor<view_const_t, u_view_const_t> checker(view);
            int errCount;
            flare::parallel_reduce("test_view_subview_const_randomaccess",
                                   flare::RangePolicy<Exec>(0, view.extent(0)), checker,
                                   errCount);
            REQUIRE_EQ(0, errCount);
        }
    }  // namespace issue_5594

    TEST_CASE("cuda, view_subview_const_randomaccess") {
        issue_5594::test_view_subview_const_randomaccess<flare::Cuda,
                flare::CudaSpace>();
        issue_5594::test_view_subview_const_randomaccess<flare::Cuda,
                flare::CudaUVMSpace>();
    }

}  // namespace Test
