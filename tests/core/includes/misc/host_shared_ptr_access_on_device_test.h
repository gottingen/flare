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

#include <flare/core/common/string_manipulation.h>
#include <flare/core/memory/host_shared_ptr.h>
#include <flare/core.h>

#include <doctest.h>

using flare::detail::HostSharedPtr;

namespace {

    class Data {
        char d[64];

    public:
        FLARE_FUNCTION void write(char const *s) {
            flare::detail::strncpy(d, s, sizeof(d));
        }
    };

    template<class SmartPtr>
    struct CheckAccessStoredPointerAndDereferenceOnDevice {
        SmartPtr m_device_ptr;
        using ElementType = typename SmartPtr::element_type;
        static_assert(std::is_same<ElementType, Data>::value, "");

        CheckAccessStoredPointerAndDereferenceOnDevice(SmartPtr device_ptr)
                : m_device_ptr(device_ptr) {
            int errors;
            flare::parallel_reduce(flare::RangePolicy<TEST_EXECSPACE>(0, 1), *this,
                                   errors);
            REQUIRE_EQ(errors, 0);
        }

        FLARE_FUNCTION void operator()(int, int &e) const {
            auto raw_ptr = m_device_ptr.get();  // get

            auto tmp = new(raw_ptr) ElementType();

            auto &obj = *m_device_ptr;  // operator*
            if (&obj != raw_ptr) ++e;

            m_device_ptr->write("hello world");  // operator->

            tmp->~ElementType();
        }
    };

    template<class Ptr>
    CheckAccessStoredPointerAndDereferenceOnDevice<Ptr>
    check_access_stored_pointer_and_dereference_on_device(Ptr p) {
        return {p};
    }

    template<class SmartPtr>
    struct CheckSpecialMembersOnDevice {
        SmartPtr m_device_ptr;

        FLARE_FUNCTION void operator()(int, int &e) const {
            SmartPtr p1 = m_device_ptr;   // copy construction
            SmartPtr p2 = std::move(p1);  // move construction

            p1 = p2;             // copy assignment
            p2 = std::move(p1);  // move assignment

            SmartPtr p3;  // default constructor
            if (p3) ++e;
            SmartPtr p4{nullptr};
            if (p4) ++e;
        }

        CheckSpecialMembersOnDevice(SmartPtr device_ptr) : m_device_ptr(device_ptr) {
            int errors;
            flare::parallel_reduce(flare::RangePolicy<TEST_EXECSPACE>(0, 1), *this,
                                   errors);
            REQUIRE_EQ(errors, 0);
        }
    };

    template<class Ptr>
    CheckSpecialMembersOnDevice<Ptr> check_special_members_on_device(Ptr p) {
        return {p};
    }

}  // namespace

TEST_CASE("TEST_CATEGORY, host_shared_ptr_dereference_on_device") {
    using T = Data;

    using MemorySpace = TEST_EXECSPACE::memory_space;

    HostSharedPtr<T> device_ptr(
            static_cast<T *>(flare::flare_malloc<MemorySpace>(sizeof(T))),
            [](T *p) { flare::flare_free<MemorySpace>(p); });

    check_access_stored_pointer_and_dereference_on_device(device_ptr);
}


TEST_CASE("TEST_CATEGORY, host_shared_ptr_special_members_on_device") {
    using T = Data;

    using MemorySpace = TEST_EXECSPACE::memory_space;

    HostSharedPtr<T> device_ptr(
            static_cast<T *>(flare::flare_malloc<MemorySpace>(sizeof(T))),
            [](T *p) { flare::flare_free<MemorySpace>(p); });

    check_special_members_on_device(device_ptr);
}

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
namespace {

    struct Bar {
        double val;
    };

    struct Foo {
        Foo(bool allocate = false) : ptr(allocate ? new Bar : nullptr) {}

        flare::detail::HostSharedPtr<Bar> ptr;

        int use_count() { return ptr.use_count(); }
    };

    template<class DevMemSpace, class HostMemSpace>
    void host_shared_ptr_test_reference_counting() {
        using ExecSpace = typename DevMemSpace::execution_space;
        bool is_gpu =
                !flare::SpaceAccessibility<ExecSpace, flare::HostSpace>::accessible;

        // Create two tracked instances
        Foo f1(true), f2(true);
        // Scope Tensors
        {
            Foo *fp_d_ptr =
                    static_cast<Foo *>(flare::flare_malloc<DevMemSpace>(sizeof(Foo)));
            flare::Tensor<Foo, DevMemSpace> fp_d(fp_d_ptr);
            // If using UVM or on the CPU don't make an extra HostCopy
            Foo *fp_h_ptr = std::is_same<DevMemSpace, HostMemSpace>::value
                            ? fp_d_ptr
                            : static_cast<Foo *>(
                                    flare::flare_malloc<HostMemSpace>(sizeof(Foo)));
            flare::Tensor<Foo, HostMemSpace> fp_h(fp_h_ptr);
            REQUIRE_EQ(1, f1.use_count());
            REQUIRE_EQ(1, f2.use_count());

            // Just for the sake of it initialize the data of the host copy
            new(fp_h.data()) Foo();
            // placement new in kernel
            //  if on GPU: should not increase use_count, fp_d will not be tracked
            //  if on Host: refcount will increase fp_d is tracked
            flare::parallel_for(
                    flare::RangePolicy<ExecSpace>(0, 1),
                    FLARE_LAMBDA(int) { new(fp_d.data()) Foo(f1); });
            flare::fence();
            flare::deep_copy(fp_h, fp_d);

            if (is_gpu)
                REQUIRE_EQ(1, f1.use_count());
            else
                REQUIRE_EQ(2, f1.use_count());
            REQUIRE_EQ(1, f2.use_count());

            // assignment operator on host, will increase f2 use_count
            //   if default device is GPU: fp_h was untracked
            //   if default device is CPU: fp_h was tracked and use_count was 2 for
            //   aliasing f1, in which case use_count will be decreased here
            fp_h() = f2;
            REQUIRE_EQ(1, f1.use_count());
            REQUIRE_EQ(2, f2.use_count());

            flare::deep_copy(fp_d, fp_h);
            REQUIRE_EQ(1, f1.use_count());
            REQUIRE_EQ(2, f2.use_count());

            // assignment in kernel:
            //  If on GPU: should not increase use_count of f1 and fp_d will not be
            //  tracked.
            //  If on Host: use_count will increase of f1, fp_d is tracked,
            //  use_count of f2 goes down.
            //  Since we are messing with the use count on the device: make host copy
            //  untracked first. Note if fp_d and fp_h alias each other (e.g. compiling
            //  for CPU only) that means fp_d() will be untracked too during assignemnt
            fp_h() = Foo();
            flare::parallel_for(
                    flare::RangePolicy<ExecSpace>(0, 1),
                    FLARE_LAMBDA(int) { fp_d() = f1; });
            flare::fence();
            flare::deep_copy(fp_h, fp_d);

            if (is_gpu)
                REQUIRE_EQ(1, f1.use_count());
            else
                REQUIRE_EQ(2, f1.use_count());
            REQUIRE_EQ(1, f2.use_count());

            // Assign non-tracked ptr
            //   if  if_gpu will not change use_count
            //   if !is_gpu will decrease use_count of f1
            fp_h() = Foo();
            REQUIRE_EQ(1, f1.use_count());
            REQUIRE_EQ(1, f2.use_count());
            fp_h() = f2;
            REQUIRE_EQ(1, f1.use_count());
            REQUIRE_EQ(2, f2.use_count());

            // before deleting host version make sure its not tracked
            fp_h() = Foo();
            if (fp_h_ptr != fp_d_ptr) flare::flare_free<HostMemSpace>(fp_h_ptr);
            flare::flare_free<DevMemSpace>(fp_d_ptr);
        }

        REQUIRE_EQ(1, f1.use_count());
        REQUIRE_EQ(1, f2.use_count());
    }
}  // namespace

TEST_CASE("TEST_CATEGORY, host_shared_ptr_tracking") {
    host_shared_ptr_test_reference_counting<typename TEST_EXECSPACE::memory_space,
            flare::HostSpace>();
#ifdef FLARE_ON_CUDA_DEVICE
    if (std::is_same<TEST_EXECSPACE, flare::Cuda>::value)
      host_shared_ptr_test_reference_counting<flare::CudaUVMSpace,
                                              flare::CudaUVMSpace>();
#endif
}

#endif  // FLARE_ENABLE_CXX11_DISPATCH_LAMBDA
