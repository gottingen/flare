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

/*--------------------------------------------------------------------------*/

namespace Test {

    struct SharedAllocDestroy {
        volatile int *count;

        SharedAllocDestroy() = default;

        SharedAllocDestroy(int *arg) : count(arg) {}

        void destroy_shared_allocation() { flare::atomic_increment(count); }
    };

    template<class MemorySpace, class ExecutionSpace>
    void test_shared_alloc() {
        using Header = const flare::detail::SharedAllocationHeader;
        using Tracker = flare::detail::SharedAllocationTracker;
        using RecordBase = flare::detail::SharedAllocationRecord<void, void>;
        using RecordMemS = flare::detail::SharedAllocationRecord<MemorySpace, void>;
        using RecordFull =
                flare::detail::SharedAllocationRecord<MemorySpace, SharedAllocDestroy>;

        static_assert(sizeof(Tracker) == sizeof(int *),
                      "SharedAllocationTracker has wrong size!");

        MemorySpace s;

        const size_t N = 1200;
        const size_t size = 8;

        RecordMemS *rarray[N];
        Header *harray[N];

        RecordMemS **const r = rarray;
        Header **const h = harray;

        flare::RangePolicy<ExecutionSpace> range(0, N);

        {
            // Since always executed on host space, leave [=]
            flare::parallel_for(range, [=](int i) {
                char name[64];
                snprintf(name, 64, "test_%.2d", i);

                r[i] = RecordMemS::allocate(s, name, size * (i + 1));
                h[i] = Header::get_header(r[i]->data());

                REQUIRE_EQ(r[i]->use_count(), 0);

                for (int j = 0; j < (i / 10) + 1; ++j) RecordBase::increment(r[i]);

                REQUIRE_EQ(r[i]->use_count(), (i / 10) + 1);
                REQUIRE_EQ(r[i], RecordMemS::get_record(r[i]->data()));
            });

            flare::fence();

#ifdef FLARE_ENABLE_DEBUG
            // Sanity check for the whole set of allocation records to which this record
            // belongs.
            RecordBase::is_sane(r[0]);
            // RecordMemS::print_records( std::cout, s, true );
#endif

            // This must be a plain for-loop since deallocation (which can be triggered
            // by RecordBase::decrement) fences all execution space instances. If this
            // is a parallel_for, the test can hang with the parallel_for blocking
            // waiting for itself to complete.
            for (size_t i = range.begin(); i < range.end(); ++i) {
                while (nullptr !=
                       (r[i] = static_cast<RecordMemS *>(RecordBase::decrement(r[i])))) {
#ifdef FLARE_ENABLE_DEBUG
                    if (r[i]->use_count() == 1) RecordBase::is_sane(r[i]);
#endif
                }
            }

            flare::fence();
        }

        {
            int destroy_count = 0;
            SharedAllocDestroy counter(&destroy_count);

            flare::parallel_for(range, [=](size_t i) {
                char name[64];
                snprintf(name, 64, "test_%.2d", int(i));

                RecordFull *rec = RecordFull::allocate(s, name, size * (i + 1));

                rec->m_destroy = counter;

                r[i] = rec;
                h[i] = Header::get_header(r[i]->data());

                REQUIRE_EQ(r[i]->use_count(), 0);

                for (size_t j = 0; j < (i / 10) + 1; ++j) RecordBase::increment(r[i]);

                REQUIRE_EQ(r[i]->use_count(), int((i / 10) + 1));
                REQUIRE_EQ(r[i], RecordMemS::get_record(r[i]->data()));
            });

            flare::fence();

#ifdef FLARE_ENABLE_DEBUG
            RecordBase::is_sane(r[0]);
#endif

            // This must be a plain for-loop since deallocation (which can be triggered
            // by RecordBase::decrement) fences all execution space instances. If this
            // is a parallel_for, the test can hang with the parallel_for blocking
            // waiting for itself to complete.
            for (size_t i = range.begin(); i < range.end(); ++i) {
                while (nullptr !=
                       (r[i] = static_cast<RecordMemS *>(RecordBase::decrement(r[i])))) {
#ifdef FLARE_ENABLE_DEBUG
                    if (r[i]->use_count() == 1) RecordBase::is_sane(r[i]);
#endif
                }
            }

            flare::fence();

            REQUIRE_EQ(destroy_count, int(N));
        }

        {
            int destroy_count = 0;

            {
                RecordFull *rec = RecordFull::allocate(s, "test", size);

                // ... Construction of the allocated { rec->data(), rec->size() }

                // Copy destruction function object into the allocation record.
                rec->m_destroy = SharedAllocDestroy(&destroy_count);

                REQUIRE_EQ(rec->use_count(), 0);

                // Start tracking, increments the use count from 0 to 1.
                Tracker track;

                track.assign_allocated_record_to_uninitialized(rec);

                REQUIRE_EQ(rec->use_count(), 1);
                REQUIRE_EQ(track.use_count(), 1);

                // Verify construction / destruction increment.
                for (size_t i = 0; i < N; ++i) {
                    REQUIRE_EQ(rec->use_count(), 1);

                    {
                        Tracker local_tracker;
                        local_tracker.assign_allocated_record_to_uninitialized(rec);
                        REQUIRE_EQ(rec->use_count(), 2);
                        REQUIRE_EQ(local_tracker.use_count(), 2);
                    }

                    REQUIRE_EQ(rec->use_count(), 1);
                    REQUIRE_EQ(track.use_count(), 1);
                }

                flare::parallel_for(range, [=](size_t) {
                    Tracker local_tracker;
                    local_tracker.assign_allocated_record_to_uninitialized(rec);
                    REQUIRE_GT(rec->use_count(), 1);
                });

                flare::fence();

                REQUIRE_EQ(rec->use_count(), 1);
                REQUIRE_EQ(track.use_count(), 1);

                // Destruction of 'track' object deallocates the 'rec' and invokes the
                // destroy function object.
            }

            REQUIRE_EQ(destroy_count, 1);
        }
    }

    TEST_CASE("TEST_CATEGORY, impl_shared_alloc") {
#ifdef TEST_CATEGORY_NUMBER
#if (TEST_CATEGORY_NUMBER < 4)  // serial threads openmp

    test_shared_alloc<flare::HostSpace, TEST_EXECSPACE>();

#elif (TEST_CATEGORY_NUMBER == 4)  // cuda
    test_shared_alloc<flare::CudaSpace, flare::DefaultHostExecutionSpace>();
#endif
#else
    test_shared_alloc<TEST_EXECSPACE, flare::DefaultHostExecutionSpace>();
#endif
}

}  // namespace Test
