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

#ifndef FLARE_BACKEND_CUDA_CUDA_TEAM_H_
#define FLARE_BACKEND_CUDA_CUDA_TEAM_H_

#include <algorithm>

#include <flare/core/defines.h>

/* only compile this file if CUDA is enabled for flare */
#if defined(FLARE_ON_CUDA_DEVICE)

#include <utility>
#include <flare/core/parallel/parallel.h>

#include <flare/backend/cuda/cuda_kernel_launch.h>
#include <flare/backend/cuda/cuda_reduce_scan.h>
#include <flare/backend/cuda/cuda_block_size_deduction.h>
#include <flare/core/tensor/vectorization.h>

#include <flare/core/profile/tools.h>
#include <typeinfo>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare::detail {

    template<typename Type>
    struct CudaJoinFunctor {
        using value_type = Type;

        FLARE_INLINE_FUNCTION
        static void join(value_type &update, const value_type &input) {
            update += input;
        }
    };

/**\brief  Team member_type passed to TeamPolicy or TeamTask closures.
 *
 *  Cuda thread blocks for team closures are dimensioned as:
 *    blockDim.x == number of "vector lanes" per "thread"
 *    blockDim.y == number of "threads" per team
 *    blockDim.z == number of teams in a block
 *  where
 *    A set of teams exactly fill a warp OR a team is the whole block
 *      ( 0 == WarpSize % ( blockDim.x * blockDim.y ) )
 *      OR
 *      ( 1 == blockDim.z )
 *
 *  Thus when 1 < blockDim.z the team is warp-synchronous
 *  and __syncthreads should not be called in team collectives.
 *
 *  When multiple teams are mapped onto a single block then the
 *  total available shared memory must be partitioned among teams.
 */
    class CudaTeamMember {
    public:
        using execution_space = flare::Cuda;
        using scratch_memory_space = execution_space::scratch_memory_space;
        using team_handle = CudaTeamMember;

    private:
        mutable void *m_team_reduce;
        scratch_memory_space m_team_shared;
        int m_team_reduce_size;
        int m_league_rank;
        int m_league_size;

    public:
        FLARE_INLINE_FUNCTION
        const execution_space::scratch_memory_space &team_shmem() const {
            return m_team_shared.set_team_thread_mode(0, 1, 0);
        }

        FLARE_INLINE_FUNCTION
        const execution_space::scratch_memory_space &team_scratch(
                const int &level) const {
            return m_team_shared.set_team_thread_mode(level, 1, 0);
        }

        FLARE_INLINE_FUNCTION
        const execution_space::scratch_memory_space &thread_scratch(
                const int &level) const {
            return m_team_shared.set_team_thread_mode(level, team_size(), team_rank());
        }

        FLARE_INLINE_FUNCTION int league_rank() const { return m_league_rank; }

        FLARE_INLINE_FUNCTION int league_size() const { return m_league_size; }

        FLARE_INLINE_FUNCTION int team_rank() const {
            FLARE_IF_ON_DEVICE((return threadIdx.y;))
            FLARE_IF_ON_HOST((return 0;))
        }

        FLARE_INLINE_FUNCTION int team_size() const {
            FLARE_IF_ON_DEVICE((return blockDim.y;))
            FLARE_IF_ON_HOST((return 1;))
        }

        FLARE_INLINE_FUNCTION void team_barrier() const {
            FLARE_IF_ON_DEVICE((
                                       if (1 == blockDim.z) { __syncthreads(); }  // team == block
                                       else { __threadfence_block(); }            // team <= warp
                               ))
        }

        //--------------------------------------------------------------------------

        template<class ValueType>
        FLARE_INLINE_FUNCTION void team_broadcast(ValueType &val,
                                                  const int &thread_id) const {
            (void) val;
            (void) thread_id;
            FLARE_IF_ON_DEVICE((
                                       if (1 == blockDim.z) {  // team == block
                                       __syncthreads();
                                       // Wait for shared data write until all threads arrive here
                                       if (threadIdx.x == 0u && threadIdx.y == (uint32_t)thread_id) {
                                       *((ValueType*)m_team_reduce) = val;
                               }
                                       __syncthreads();  // Wait for shared data read until root thread
                                       // writes
                                       val = *((ValueType*)m_team_reduce);
                               } else {               // team <= warp
                                       ValueType tmp(val);  // input might not be a register variable
                                       detail::in_place_shfl(val, tmp, blockDim.x * thread_id,
                                       blockDim.x * blockDim.y);
                               }))
        }

        template<class Closure, class ValueType>
        FLARE_INLINE_FUNCTION void team_broadcast(Closure const &f, ValueType &val,
                                                  const int &thread_id) const {
            (void) f;
            (void) val;
            (void) thread_id;
            FLARE_IF_ON_DEVICE((
                                       f(val);

                                       if (1 == blockDim.z) {  // team == block
                                           __syncthreads();
                                           // Wait for shared data write until all threads arrive here
                                           if (threadIdx.x == 0u && threadIdx.y == (uint32_t) thread_id) {
                                               *((ValueType *) m_team_reduce) = val;
                                           }
                                           __syncthreads();  // Wait for shared data read until root thread
                                           // writes
                                           val = *((ValueType *) m_team_reduce);
                                       } else {               // team <= warp
                                           ValueType tmp(val);  // input might not be a register variable
                                           detail::in_place_shfl(val, tmp, blockDim.x * thread_id,
                                                                 blockDim.x * blockDim.y);
                                       }))
        }

        //--------------------------------------------------------------------------
        /**\brief  Reduction across a team
         *
         *  Mapping of teams onto blocks:
         *    blockDim.x  is "vector lanes"
         *    blockDim.y  is team "threads"
         *    blockDim.z  is number of teams per block
         *
         *  Requires:
         *    blockDim.x is power two
         *    blockDim.x <= CudaTraits::WarpSize
         *    ( 0 == CudaTraits::WarpSize % ( blockDim.x * blockDim.y )
         *      OR
         *    ( 1 == blockDim.z )
         */
        template<typename ReducerType>
        FLARE_INLINE_FUNCTION std::enable_if_t<is_reducer<ReducerType>::value>
        team_reduce(ReducerType const &reducer) const noexcept {
            team_reduce(reducer, reducer.reference());
        }

        template<typename ReducerType>
        FLARE_INLINE_FUNCTION std::enable_if_t<is_reducer<ReducerType>::value>
        team_reduce(ReducerType const &reducer,
                    typename ReducerType::value_type &value) const noexcept {
            (void) reducer;
            (void) value;
            FLARE_IF_ON_DEVICE(
                    (typename detail::FunctorAnalysis<
                            detail::FunctorPatternInterface::REDUCE, TeamPolicy<Cuda>,
                            ReducerType, typename ReducerType::value_type>::Reducer
                    wrapped_reducer(reducer);
                    cuda_intra_block_reduction(value, wrapped_reducer, blockDim.y);
                    reducer.reference() = value;))
        }

        //--------------------------------------------------------------------------
        /** \brief  Intra-team exclusive prefix sum with team_rank() ordering
         *          with intra-team non-deterministic ordering accumulation.
         *
         *  The global inter-team accumulation value will, at the end of the
         *  league's parallel execution, be the scan's total.
         *  Parallel execution ordering of the league's teams is non-deterministic.
         *  As such the base value for each team's scan operation is similarly
         *  non-deterministic.
         */
        template<typename Type>
        FLARE_INLINE_FUNCTION Type team_scan(const Type &value,
                                             Type *const global_accum) const {
            FLARE_IF_ON_DEVICE((
                                       Type * const base_data = (Type *) m_team_reduce;

                                       __syncthreads();  // Don't write in to shared data until all threads
                                       // have entered this function

                                       if (0 == threadIdx.y) { base_data[0] = 0; }

                                       base_data[threadIdx.y + 1] = value;
                                       detail::CudaJoinFunctor<Type> cuda_join_functor;
                                       typename detail::FunctorAnalysis<
                                               detail::FunctorPatternInterface::SCAN, TeamPolicy<Cuda>,
                                               detail::CudaJoinFunctor<Type>, Type>::Reducer
                                               reducer(cuda_join_functor);
                                       detail::cuda_intra_block_reduce_scan<true>(reducer, base_data + 1);

                                       if (global_accum) {
                                           if (blockDim.y == threadIdx.y + 1) {
                                               base_data[blockDim.y] =
                                                       atomic_fetch_add(global_accum, base_data[blockDim.y]);
                                           }
                                           __syncthreads();  // Wait for atomic
                                           base_data[threadIdx.y] += base_data[blockDim.y];
                                       }

                                       return base_data[threadIdx.y];))

            FLARE_IF_ON_HOST(((void) value; (void) global_accum; return Type();))
        }

        /** \brief  Intra-team exclusive prefix sum with team_rank() ordering.
         *
         *  The highest rank thread can compute the reduction total as
         *    reduction_total = dev.team_scan( value ) + value ;
         */
        template<typename Type>
        FLARE_INLINE_FUNCTION Type team_scan(const Type &value) const {
            return this->template team_scan<Type>(value, nullptr);
        }

        //----------------------------------------

        template<typename ReducerType>
        FLARE_INLINE_FUNCTION static std::enable_if_t<is_reducer<ReducerType>::value>
        vector_reduce(ReducerType const &reducer) {
            vector_reduce(reducer, reducer.reference());
        }

        template<typename ReducerType>
        FLARE_INLINE_FUNCTION static std::enable_if_t<is_reducer<ReducerType>::value>
        vector_reduce(ReducerType const &reducer,
                      typename ReducerType::value_type &value) {
            (void) reducer;
            (void) value;
            FLARE_IF_ON_DEVICE(
            (if (blockDim.x == 1) return;

                    // Intra vector lane shuffle reduction:
                    typename ReducerType::value_type tmp(value);
                    typename ReducerType::value_type tmp2 = tmp;

                    unsigned mask =
                    blockDim.x == 32
                    ? 0xffffffff
                    : ((1 << blockDim.x) - 1)
                    << ((threadIdx.y % (32 / blockDim.x)) * blockDim.x);

                    for (int i = blockDim.x; (i >>= 1);) {
                    detail::in_place_shfl_down(tmp2, tmp, i, blockDim.x, mask);
                    if ((int)threadIdx.x < i) {
                    reducer.join(tmp, tmp2);
            }
            }

                    // Broadcast from root lane to all other lanes.
                    // Cannot use "butterfly" algorithm to avoid the broadcast
                    // because floating point summation is not associative
                    // and thus different threads could have different results.

                    detail::in_place_shfl(tmp2, tmp, 0, blockDim.x, mask);
                    value = tmp2; reducer.reference() = tmp2;))
        }

        //----------------------------------------
        // Private for the driver

        FLARE_INLINE_FUNCTION
        CudaTeamMember(void *shared, const size_t shared_begin,
                       const size_t shared_size, void *scratch_level_1_ptr,
                       const size_t scratch_level_1_size, const int arg_league_rank,
                       const int arg_league_size)
                : m_team_reduce(shared),
                  m_team_shared(static_cast<char *>(shared) + shared_begin, shared_size,
                                scratch_level_1_ptr, scratch_level_1_size),
                  m_team_reduce_size(shared_begin),
                  m_league_rank(arg_league_rank),
                  m_league_size(arg_league_size) {}

    public:
        // Declare to avoid unused private member warnings which are trigger
        // when SFINAE excludes the member function which uses these variables
        // Making another class a friend also surpresses these warnings
        bool impl_avoid_sfinae_warning() const noexcept {
            return m_team_reduce_size > 0 && m_team_reduce != nullptr;
        }
    };

    template<typename iType>
    struct TeamThreadRangeBoundariesStruct<iType, CudaTeamMember> {
        using index_type = iType;
        const CudaTeamMember &member;
        const iType start;
        const iType end;

        FLARE_INLINE_FUNCTION
        TeamThreadRangeBoundariesStruct(const CudaTeamMember &thread_, iType count)
                : member(thread_), start(0), end(count) {}

        FLARE_INLINE_FUNCTION
        TeamThreadRangeBoundariesStruct(const CudaTeamMember &thread_, iType begin_,
                                        iType end_)
                : member(thread_), start(begin_), end(end_) {}
    };

    template<typename iType>
    struct TeamVectorRangeBoundariesStruct<iType, CudaTeamMember> {
        using index_type = iType;
        const CudaTeamMember &member;
        const iType start;
        const iType end;

        FLARE_INLINE_FUNCTION
        TeamVectorRangeBoundariesStruct(const CudaTeamMember &thread_,
                                        const iType &count)
                : member(thread_), start(0), end(count) {}

        FLARE_INLINE_FUNCTION
        TeamVectorRangeBoundariesStruct(const CudaTeamMember &thread_,
                                        const iType &begin_, const iType &end_)
                : member(thread_), start(begin_), end(end_) {}
    };

    template<typename iType>
    struct ThreadVectorRangeBoundariesStruct<iType, CudaTeamMember> {
        using index_type = iType;
        const index_type start;
        const index_type end;

        FLARE_INLINE_FUNCTION
        ThreadVectorRangeBoundariesStruct(const CudaTeamMember, index_type count)
                : start(static_cast<index_type>(0)), end(count) {}

        FLARE_INLINE_FUNCTION
        ThreadVectorRangeBoundariesStruct(const CudaTeamMember, index_type arg_begin,
                                          index_type arg_end)
                : start(arg_begin), end(arg_end) {}
    };

}  // namespace flare::detail

namespace flare {
    template<typename iType>
    FLARE_INLINE_FUNCTION
    detail::TeamThreadRangeBoundariesStruct<iType, detail::CudaTeamMember>
    TeamThreadRange(const detail::CudaTeamMember &thread, iType count) {
        return detail::TeamThreadRangeBoundariesStruct<iType, detail::CudaTeamMember>(
                thread, count);
    }

    template<typename iType1, typename iType2>
    FLARE_INLINE_FUNCTION detail::TeamThreadRangeBoundariesStruct<
            std::common_type_t<iType1, iType2>, detail::CudaTeamMember>
    TeamThreadRange(const detail::CudaTeamMember &thread, iType1 begin, iType2 end) {
        using iType = std::common_type_t<iType1, iType2>;
        return detail::TeamThreadRangeBoundariesStruct<iType, detail::CudaTeamMember>(
                thread, iType(begin), iType(end));
    }

    template<typename iType>
    FLARE_INLINE_FUNCTION
    detail::TeamVectorRangeBoundariesStruct<iType, detail::CudaTeamMember>
    TeamVectorRange(const detail::CudaTeamMember &thread, const iType &count) {
        return detail::TeamVectorRangeBoundariesStruct<iType, detail::CudaTeamMember>(
                thread, count);
    }

    template<typename iType1, typename iType2>
    FLARE_INLINE_FUNCTION detail::TeamVectorRangeBoundariesStruct<
            std::common_type_t<iType1, iType2>, detail::CudaTeamMember>
    TeamVectorRange(const detail::CudaTeamMember &thread, const iType1 &begin,
                    const iType2 &end) {
        using iType = std::common_type_t<iType1, iType2>;
        return detail::TeamVectorRangeBoundariesStruct<iType, detail::CudaTeamMember>(
                thread, iType(begin), iType(end));
    }

    template<typename iType>
    FLARE_INLINE_FUNCTION
    detail::ThreadVectorRangeBoundariesStruct<iType, detail::CudaTeamMember>
    ThreadVectorRange(const detail::CudaTeamMember &thread, iType count) {
        return detail::ThreadVectorRangeBoundariesStruct<iType, detail::CudaTeamMember>(
                thread, count);
    }

    template<typename iType1, typename iType2>
    FLARE_INLINE_FUNCTION detail::ThreadVectorRangeBoundariesStruct<
            std::common_type_t<iType1, iType2>, detail::CudaTeamMember>
    ThreadVectorRange(const detail::CudaTeamMember &thread, iType1 arg_begin,
                      iType2 arg_end) {
        using iType = std::common_type_t<iType1, iType2>;
        return detail::ThreadVectorRangeBoundariesStruct<iType, detail::CudaTeamMember>(
                thread, iType(arg_begin), iType(arg_end));
    }

    FLARE_INLINE_FUNCTION
    detail::ThreadSingleStruct<detail::CudaTeamMember> PerTeam(
            const detail::CudaTeamMember &thread) {
        return detail::ThreadSingleStruct<detail::CudaTeamMember>(thread);
    }

    FLARE_INLINE_FUNCTION
    detail::VectorSingleStruct<detail::CudaTeamMember> PerThread(
            const detail::CudaTeamMember &thread) {
        return detail::VectorSingleStruct<detail::CudaTeamMember>(thread);
    }

//----------------------------------------------------------------------------

/** \brief  Inter-thread parallel_for.
 *
 *  Executes closure(iType i) for each i=[0..N).
 *
 * The range [0..N) is mapped to all threads of the the calling thread team.
 */
    template<typename iType, class Closure>
    FLARE_INLINE_FUNCTION void parallel_for(
            const detail::TeamThreadRangeBoundariesStruct<iType, detail::CudaTeamMember> &
            loop_boundaries,
            const Closure &closure) {
        (void) loop_boundaries;
        (void) closure;
        FLARE_IF_ON_DEVICE(
        (for (iType i = loop_boundaries.start + threadIdx.y;
                i < loop_boundaries.end; i += blockDim.y) { closure(i); }))
    }

//----------------------------------------------------------------------------

/** \brief  Inter-thread parallel_reduce with a reducer.
 *
 *  Executes closure(iType i, ValueType & val) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all threads of the
 *  calling thread team and a summation of val is
 *  performed and put into result.
 */
    template<typename iType, class Closure, class ReducerType>
    FLARE_INLINE_FUNCTION std::enable_if_t<flare::is_reducer<ReducerType>::value>
    parallel_reduce(const detail::TeamThreadRangeBoundariesStruct<
            iType, detail::CudaTeamMember> &loop_boundaries,
                    const Closure &closure, const ReducerType &reducer) {
        FLARE_IF_ON_DEVICE(
                (typename ReducerType::value_type value;

                reducer.init(value);

                for (iType i = loop_boundaries.start + threadIdx.y;
                     i < loop_boundaries.end; i += blockDim.y) { closure(i, value); }

                loop_boundaries.member.team_reduce(reducer, value);))
        // Avoid bogus warning about reducer value being uninitialized with combined
        // reducers
        FLARE_IF_ON_HOST(((void) loop_boundaries; (void) closure;
                                 reducer.init(reducer.reference());
                                 flare::abort("Should only run on the device!");));
    }

/** \brief  Inter-thread parallel_reduce assuming summation.
 *
 *  Executes closure(iType i, ValueType & val) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all threads of the
 *  calling thread team and a summation of val is
 *  performed and put into result.
 */
    template<typename iType, class Closure, typename ValueType>
    FLARE_INLINE_FUNCTION std::enable_if_t<!flare::is_reducer<ValueType>::value>
    parallel_reduce(const detail::TeamThreadRangeBoundariesStruct<
            iType, detail::CudaTeamMember> &loop_boundaries,
                    const Closure &closure, ValueType &result) {
        (void) loop_boundaries;
        (void) closure;
        (void) result;
        FLARE_IF_ON_DEVICE(
                (ValueType val; flare::Sum<ValueType> reducer(val);

                reducer.init(reducer.reference());

                for (iType i = loop_boundaries.start + threadIdx.y;
                     i < loop_boundaries.end; i += blockDim.y) { closure(i, val); }

                loop_boundaries.member.team_reduce(reducer, val);
                result = reducer.reference();))
    }

    template<typename iType, class Closure>
    FLARE_INLINE_FUNCTION void parallel_for(
            const detail::TeamVectorRangeBoundariesStruct<iType, detail::CudaTeamMember> &
            loop_boundaries,
            const Closure &closure) {
        (void) loop_boundaries;
        (void) closure;
        FLARE_IF_ON_DEVICE((for (iType i = loop_boundaries.start +
                                   threadIdx.y * blockDim.x + threadIdx.x;
                                   i < loop_boundaries.end;
                                   i += blockDim.y * blockDim.x) { closure(i); }))
    }

    template<typename iType, class Closure, class ReducerType>
    FLARE_INLINE_FUNCTION std::enable_if_t<flare::is_reducer<ReducerType>::value>
    parallel_reduce(const detail::TeamVectorRangeBoundariesStruct<
            iType, detail::CudaTeamMember> &loop_boundaries,
                    const Closure &closure, const ReducerType &reducer) {
        FLARE_IF_ON_DEVICE((typename ReducerType::value_type value;
                                   reducer.init(value);

                                   for (iType i = loop_boundaries.start +
                                                  threadIdx.y * blockDim.x + threadIdx.x;
                                        i < loop_boundaries.end;
                                        i += blockDim.y * blockDim.x) { closure(i, value); }

                                   loop_boundaries.member.vector_reduce(reducer, value);
                                   loop_boundaries.member.team_reduce(reducer, value);))
        // Avoid bogus warning about reducer value being uninitialized with combined
        // reducers
        FLARE_IF_ON_HOST(((void) loop_boundaries; (void) closure;
                                 reducer.init(reducer.reference());
                                 flare::abort("Should only run on the device!");));
    }

    template<typename iType, class Closure, typename ValueType>
    FLARE_INLINE_FUNCTION std::enable_if_t<!flare::is_reducer<ValueType>::value>
    parallel_reduce(const detail::TeamVectorRangeBoundariesStruct<
            iType, detail::CudaTeamMember> &loop_boundaries,
                    const Closure &closure, ValueType &result) {
        (void) loop_boundaries;
        (void) closure;
        (void) result;
        FLARE_IF_ON_DEVICE((ValueType val; flare::Sum<ValueType> reducer(val);

                                   reducer.init(reducer.reference());

                                   for (iType i = loop_boundaries.start +
                                                  threadIdx.y * blockDim.x + threadIdx.x;
                                        i < loop_boundaries.end;
                                        i += blockDim.y * blockDim.x) { closure(i, val); }

                                   loop_boundaries.member.vector_reduce(reducer);
                                   loop_boundaries.member.team_reduce(reducer);
                                   result = reducer.reference();))
    }

//----------------------------------------------------------------------------

/** \brief  Intra-thread vector parallel_for.
 *
 *  Executes closure(iType i) for each i=[0..N)
 *
 * The range [0..N) is mapped to all vector lanes of the the calling thread.
 */
    template<typename iType, class Closure>
    FLARE_INLINE_FUNCTION void parallel_for(
            const detail::ThreadVectorRangeBoundariesStruct<iType, detail::CudaTeamMember> &
            loop_boundaries,
            const Closure &closure) {
        (void) loop_boundaries;
        (void) closure;
        FLARE_IF_ON_DEVICE((
                                   for (iType i = loop_boundaries.start + threadIdx.x;
                                   i < loop_boundaries.end; i += blockDim.x) { closure(i); }

                                   __syncwarp(blockDim.x == 32
                                   ? 0xffffffff
                                   : ((1 << blockDim.x) - 1)
                                   << (threadIdx.y % (32 / blockDim.x)) * blockDim.x);))
    }

//----------------------------------------------------------------------------

/** \brief  Intra-thread vector parallel_reduce.
 *
 *  Calls closure(iType i, ValueType & val) for each i=[0..N).
 *
 *  The range [0..N) is mapped to all vector lanes of
 *  the calling thread and a reduction of val is performed using +=
 *  and output into result.
 *
 *  The identity value for the += operator is assumed to be the default
 *  constructed value.
 */
    template<typename iType, class Closure, class ReducerType>
    FLARE_INLINE_FUNCTION std::enable_if_t<is_reducer<ReducerType>::value>
    parallel_reduce(detail::ThreadVectorRangeBoundariesStruct<
            iType, detail::CudaTeamMember> const &loop_boundaries,
                    Closure const &closure, ReducerType const &reducer) {
        FLARE_IF_ON_DEVICE((

                                   reducer.init(reducer.reference());

                                   for (iType i = loop_boundaries.start + threadIdx.x;
                                        i < loop_boundaries.end;
                                        i += blockDim.x) { closure(i, reducer.reference()); }

                                   detail::CudaTeamMember::vector_reduce(reducer);

                           ))
        // Avoid bogus warning about reducer value being uninitialized with combined
        // reducers
        FLARE_IF_ON_HOST(((void) loop_boundaries; (void) closure;
                                 reducer.init(reducer.reference());
                                 flare::abort("Should only run on the device!");));
    }

/** \brief  Intra-thread vector parallel_reduce.
 *
 *  Calls closure(iType i, ValueType & val) for each i=[0..N).
 *
 *  The range [0..N) is mapped to all vector lanes of
 *  the calling thread and a reduction of val is performed using +=
 *  and output into result.
 *
 *  The identity value for the += operator is assumed to be the default
 *  constructed value.
 */
    template<typename iType, class Closure, typename ValueType>
    FLARE_INLINE_FUNCTION std::enable_if_t<!is_reducer<ValueType>::value>
    parallel_reduce(detail::ThreadVectorRangeBoundariesStruct<
            iType, detail::CudaTeamMember> const &loop_boundaries,
                    Closure const &closure, ValueType &result) {
        (void) loop_boundaries;
        (void) closure;
        (void) result;
        FLARE_IF_ON_DEVICE(
                (result = ValueType();

                for (iType i = loop_boundaries.start + threadIdx.x;
                     i < loop_boundaries.end; i += blockDim.x) { closure(i, result); }

                detail::CudaTeamMember::vector_reduce(flare::Sum<ValueType>(result));

        ))
    }

//----------------------------------------------------------------------------

/** \brief  Inter-thread parallel exclusive prefix sum.
 *
 *  Executes closure(iType i, ValueType & val, bool final) for each i=[0..N)
 *
 *  The range [0..N) is mapped to each rank in the team (whose global rank is
 *  less than N) and a scan operation is performed. The last call to closure has
 *  final == true.
 */

    template<typename iType, typename FunctorType>
    FLARE_INLINE_FUNCTION void parallel_scan(
            const detail::TeamThreadRangeBoundariesStruct<iType, detail::CudaTeamMember> &
            loop_bounds,
            const FunctorType &lambda) {
        // Extract value_type from lambda
        using value_type = typename flare::detail::FunctorAnalysis<
                flare::detail::FunctorPatternInterface::SCAN, void, FunctorType,
                void>::value_type;

        const auto start = loop_bounds.start;
        const auto end = loop_bounds.end;
        auto &member = loop_bounds.member;
        const auto team_size = member.team_size();
        const auto team_rank = member.team_rank();
        const auto nchunk = (end - start + team_size - 1) / team_size;
        value_type accum = 0;
        // each team has to process one or more chunks of the prefix scan
        for (iType i = 0; i < nchunk; ++i) {
            auto ii = start + i * team_size + team_rank;
            // local accumulation for this chunk
            value_type local_accum = 0;
            // user updates value with prefix value
            if (ii < loop_bounds.end) lambda(ii, local_accum, false);
            // perform team scan
            local_accum = member.team_scan(local_accum);
            // add this blocks accum to total accumulation
            auto val = accum + local_accum;
            // user updates their data with total accumulation
            if (ii < loop_bounds.end) lambda(ii, val, true);
            // the last value needs to be propogated to next chunk
            if (team_rank == team_size - 1) accum = val;
            // broadcast last value to rest of the team
            member.team_broadcast(accum, team_size - 1);
        }
    }

//----------------------------------------------------------------------------

/** \brief  Intra-thread vector parallel scan with reducer.
 *
 *  Executes closure(iType i, ValueType & val, bool final) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all vector lanes in the
 *  thread and a scan operation is performed.
 *  The last call to closure has final == true.
 */
    template<typename iType, class Closure, typename ReducerType>
    FLARE_INLINE_FUNCTION std::enable_if_t<flare::is_reducer<ReducerType>::value>
    parallel_scan(const detail::ThreadVectorRangeBoundariesStruct<
            iType, detail::CudaTeamMember> &loop_boundaries,
                  const Closure &closure, const ReducerType &reducer) {
        (void) loop_boundaries;
        (void) closure;
        (void) reducer;
        FLARE_IF_ON_DEVICE((

                                   using value_type = typename ReducerType::value_type;

                                   value_type accum;

                                   reducer.init(accum);

                                   const value_type identity = accum;

                                   // Loop through boundaries by vector-length chunks
                                   // must scan at each iteration

                                   // All thread "lanes" must loop the same number of times.
                                   // Determine an loop end for all thread "lanes."
                                   // Requires:
                                   //   blockDim.x is power of two and thus
                                   //     ( end % blockDim.x ) == ( end & ( blockDim.x - 1 ) )
                                   //   1 <= blockDim.x <= CudaTraits::WarpSize

                                   const int mask = blockDim.x - 1;
                                   const unsigned active_mask =
                                   blockDim.x == 32
                                   ? 0xffffffff
                                   : ((1 << blockDim.x) - 1)
                                   << (threadIdx.y % (32 / blockDim.x)) * blockDim.x;
                                   const int rem = loop_boundaries.end & mask;  // == end % blockDim.x
                                   const int end = loop_boundaries.end + (rem ? blockDim.x - rem : 0);

                                   for (int i = threadIdx.x; i < end; i += blockDim.x) {
                                   value_type val = identity;

                                   // First acquire per-lane contributions.
                                   // This sets i's val to i-1's contribution
                                   // to make the latter in_place_shfl_up an
                                   // exclusive scan -- the final accumulation
                                   // of i's val will be included in the second
                                   // closure call later.
                                   if (i < loop_boundaries.end && threadIdx.x > 0) {
                                   closure(i - 1, val, false);
                           }

                                   // Bottom up exclusive scan in triangular pattern
                                   // where each CUDA thread is the root of a reduction tree
                                   // from the zeroth "lane" to itself.
                                   //  [t] += [t-1] if t >= 1
                                   //  [t] += [t-2] if t >= 2
                                   //  [t] += [t-4] if t >= 4
                                   //  ...
                                   //  This differs from the non-reducer overload, where an inclusive scan
                                   //  was implemented, because in general the binary operator cannot be
                                   //  inverted and we would not be able to remove the inclusive
                                   //  contribution by inversion.
                                   for (int j = 1; j < (int)blockDim.x; j <<= 1) {
                                   value_type tmp = identity;
                                   detail::in_place_shfl_up(tmp, val, j, blockDim.x, active_mask);
                                   if (j <= (int)threadIdx.x) {
                                   reducer.join(val, tmp);
                           }
                           }

                                   // Include accumulation
                                   reducer.join(val, accum);

                                   // Update i's contribution into the val
                                   // and add it to accum for next round
                                   if (i < loop_boundaries.end) closure(i, val, true);
                                   detail::in_place_shfl(accum, val, mask, blockDim.x, active_mask);
                           }

                           ))
    }

    /** \brief  Intra-thread vector parallel exclusive prefix sum.
     *
     *  Executes closure(iType i, ValueType & val, bool final) for each i=[0..N)
     *
     *  The range [0..N) is mapped to all vector lanes in the
     *  thread and a scan operation is performed.
     *  The last call to closure has final == true.
     */
    template<typename iType, class Closure>
    FLARE_INLINE_FUNCTION void parallel_scan(
            const detail::ThreadVectorRangeBoundariesStruct<iType, detail::CudaTeamMember> &
            loop_boundaries,
            const Closure &closure) {
        using value_type = typename flare::detail::FunctorAnalysis<
                flare::detail::FunctorPatternInterface::SCAN, void, Closure,
                void>::value_type;
        value_type dummy;
        parallel_scan(loop_boundaries, closure, flare::Sum<value_type>(dummy));
    }

}  // namespace flare

namespace flare {

    template<class FunctorType>
    FLARE_INLINE_FUNCTION void single(
            const detail::VectorSingleStruct<detail::CudaTeamMember> &,
            const FunctorType &lambda) {
        (void) lambda;
        FLARE_IF_ON_DEVICE((
                                   if (threadIdx.x == 0) { lambda(); }

                                   __syncwarp(blockDim.x == 32
                                   ? 0xffffffff
                                   : ((1 << blockDim.x) - 1)
                                   << (threadIdx.y % (32 / blockDim.x)) * blockDim.x);))
    }

    template<class FunctorType>
    FLARE_INLINE_FUNCTION void single(
            const detail::ThreadSingleStruct<detail::CudaTeamMember> &,
            const FunctorType &lambda) {
        (void) lambda;
        FLARE_IF_ON_DEVICE((
                                   if (threadIdx.x == 0 && threadIdx.y == 0) { lambda(); }

                                   __syncwarp(blockDim.x == 32
                                   ? 0xffffffff
                                   : ((1 << blockDim.x) - 1)
                                   << (threadIdx.y % (32 / blockDim.x)) * blockDim.x);))
    }

    template<class FunctorType, class ValueType>
    FLARE_INLINE_FUNCTION void single(
            const detail::VectorSingleStruct<detail::CudaTeamMember> &,
            const FunctorType &lambda, ValueType &val) {
        (void) lambda;
        (void) val;
        FLARE_IF_ON_DEVICE(
        (if (threadIdx.x == 0) { lambda(val); }

                unsigned mask =
                blockDim.x == 32
                ? 0xffffffff
                : ((1 << blockDim.x) - 1)
                << ((threadIdx.y % (32 / blockDim.x)) * blockDim.x);

                detail::in_place_shfl(val, val, 0, blockDim.x, mask);))
    }

    template<class FunctorType, class ValueType>
    FLARE_INLINE_FUNCTION void single(
            const detail::ThreadSingleStruct<detail::CudaTeamMember> &single_struct,
            const FunctorType &lambda, ValueType &val) {
        (void) single_struct;
        (void) lambda;
        (void) val;
        FLARE_IF_ON_DEVICE(
        (if (threadIdx.x == 0 && threadIdx.y == 0) { lambda(val); }

                single_struct.team_member.team_broadcast(val, 0);))
    }

}  // namespace flare

#endif  // FLARE_ON_CUDA_DEVICE

#endif  // FLARE_BACKEND_CUDA_CUDA_TEAM_H_
