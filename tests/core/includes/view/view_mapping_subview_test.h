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

#include <doctest.h>
#include <sstream>
#include <iostream>

#include <flare/core.h>

namespace Test {

    template<class Space>
    struct TestViewMappingSubview {
        using ExecSpace = typename Space::execution_space;
        using MemSpace = typename Space::memory_space;

        using range = flare::pair<int, int>;

        enum {
            AN = 10
        };
        using AT = flare::View<int *, ExecSpace>;
        using ACT = flare::View<const int *, ExecSpace>;
        using AS = flare::Subview<AT, range>;

        enum {
            BN0 = 10, BN1 = 11, BN2 = 12
        };
        using BT = flare::View<int ***, ExecSpace>;
        using BS = flare::Subview<BT, range, range, range>;

        enum {
            CN0 = 10, CN1 = 11, CN2 = 12
        };
        using CT = flare::View<int ***[13][14], ExecSpace>;
        // changing CS to CTS here because when compiling with nvshmem, there is a
        // define for CS that makes this fail...
        using CTS = flare::Subview<CT, range, range, range, int, int>;

        enum {
            DN0 = 10, DN1 = 11, DN2 = 12, DN3 = 13, DN4 = 14
        };
        using DT = flare::View<int ***[DN3][DN4], ExecSpace>;
        using DS = flare::Subview<DT, int, range, range, range, int>;

        using DLT = flare::View<int ***[13][14], flare::LayoutLeft, ExecSpace>;
        using DLS1 = flare::Subview<DLT, range, int, int, int, int>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        static_assert(
            DLS1::rank == 1 &&
                std::is_same<typename DLS1::array_layout, flare::LayoutLeft>::value,
            "Subview layout error for rank 1 subview of left-most range of "
            "LayoutLeft");
#endif

        using DRT = flare::View<int ***[13][14], flare::LayoutRight, ExecSpace>;
        using DRS1 = flare::Subview<DRT, int, int, int, int, range>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        static_assert(
            DRS1::rank == 1 &&
                std::is_same<typename DRS1::array_layout, flare::LayoutRight>::value,
            "Subview layout error for rank 1 subview of right-most range of "
            "LayoutRight");
#endif

        AT Aa;
        AS Ab;
        ACT Ac;
        BT Ba;
        BS Bb;
        CT Ca;
        CTS Cb;
        DT Da;
        DS Db;

        TestViewMappingSubview()
                : Aa("Aa", AN),
                  Ab(flare::subview(Aa, std::pair<int, int>(1, AN - 1))),
                  Ac(Aa, std::pair<int, int>(1, AN - 1)),
                  Ba("Ba", BN0, BN1, BN2),
                  Bb(flare::subview(Ba, std::pair<int, int>(1, BN0 - 1),
                                    std::pair<int, int>(1, BN1 - 1),
                                    std::pair<int, int>(1, BN2 - 1))),
                  Ca("Ca", CN0, CN1, CN2),
                  Cb(flare::subview(Ca, std::pair<int, int>(1, CN0 - 1),
                                    std::pair<int, int>(1, CN1 - 1),
                                    std::pair<int, int>(1, CN2 - 1), 1, 2)),
                  Da("Da", DN0, DN1, DN2),
                  Db(flare::subview(Da, 1, std::pair<int, int>(1, DN1 - 1),
                                    std::pair<int, int>(1, DN2 - 1),
                                    std::pair<int, int>(1, DN3 - 1), 2)) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int, long &error_count) const {
            auto Ad = flare::subview(Aa, flare::pair<int, int>(1, AN - 1));

            for (int i = 1; i < AN - 1; ++i)
                if (&Aa[i] != &Ab[i - 1]) ++error_count;
            for (int i = 1; i < AN - 1; ++i)
                if (&Aa[i] != &Ac[i - 1]) ++error_count;
            for (int i = 1; i < AN - 1; ++i)
                if (&Aa[i] != &Ad[i - 1]) ++error_count;

            for (int i2 = 1; i2 < BN2 - 1; ++i2)
                for (int i1 = 1; i1 < BN1 - 1; ++i1)
                    for (int i0 = 1; i0 < BN0 - 1; ++i0) {
                        if (&Ba(i0, i1, i2) != &Bb(i0 - 1, i1 - 1, i2 - 1)) ++error_count;
                    }

            for (int i2 = 1; i2 < CN2 - 1; ++i2)
                for (int i1 = 1; i1 < CN1 - 1; ++i1)
                    for (int i0 = 1; i0 < CN0 - 1; ++i0) {
                        if (&Ca(i0, i1, i2, 1, 2) != &Cb(i0 - 1, i1 - 1, i2 - 1))
                            ++error_count;
                    }

            for (int i2 = 1; i2 < DN3 - 1; ++i2)
                for (int i1 = 1; i1 < DN2 - 1; ++i1)
                    for (int i0 = 1; i0 < DN1 - 1; ++i0) {
                        if (&Da(1, i0, i1, i2, 2) != &Db(i0 - 1, i1 - 1, i2 - 1))
                            ++error_count;
                    }
        }

        void run() {
            TestViewMappingSubview<ExecSpace> self;

            REQUIRE_EQ(Aa.extent(0), AN);
            REQUIRE_EQ(Ab.extent(0), (size_t) AN - 2);
            REQUIRE_EQ(Ac.extent(0), (size_t) AN - 2);
            REQUIRE_EQ(Ba.extent(0), BN0);
            REQUIRE_EQ(Ba.extent(1), BN1);
            REQUIRE_EQ(Ba.extent(2), BN2);
            REQUIRE_EQ(Bb.extent(0), (size_t) BN0 - 2);
            REQUIRE_EQ(Bb.extent(1), (size_t) BN1 - 2);
            REQUIRE_EQ(Bb.extent(2), (size_t) BN2 - 2);

            REQUIRE_EQ(Ca.extent(0), CN0);
            REQUIRE_EQ(Ca.extent(1), CN1);
            REQUIRE_EQ(Ca.extent(2), CN2);
            REQUIRE_EQ(Ca.extent(3), (size_t) 13);
            REQUIRE_EQ(Ca.extent(4), (size_t) 14);
            REQUIRE_EQ(Cb.extent(0), (size_t) CN0 - 2);
            REQUIRE_EQ(Cb.extent(1), (size_t) CN1 - 2);
            REQUIRE_EQ(Cb.extent(2), (size_t) CN2 - 2);

            REQUIRE_EQ(Da.extent(0), DN0);
            REQUIRE_EQ(Da.extent(1), DN1);
            REQUIRE_EQ(Da.extent(2), DN2);
            REQUIRE_EQ(Da.extent(3), DN3);
            REQUIRE_EQ(Da.extent(4), DN4);

            REQUIRE_EQ(Db.extent(0), (size_t) DN1 - 2);
            REQUIRE_EQ(Db.extent(1), (size_t) DN2 - 2);
            REQUIRE_EQ(Db.extent(2), (size_t) DN3 - 2);

            REQUIRE_EQ(Da.stride_1(), Db.stride_0());
            REQUIRE_EQ(Da.stride_2(), Db.stride_1());
            REQUIRE_EQ(Da.stride_3(), Db.stride_2());

            long error_count = -1;
            flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, 1), *this,
                                   error_count);

            REQUIRE_EQ(error_count, 0);
        }
    };

    TEST_CASE("TEST_CATEGORY, view_mapping_subview") {
        TestViewMappingSubview<TEST_EXECSPACE> f;
        f.run();
    }

}  // namespace Test
