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

#ifndef FLARE_CORE_POLICY_EXEC_POLICY_H_
#define FLARE_CORE_POLICY_EXEC_POLICY_H_

#include <flare/core_fwd.h>
#include <flare/core/common/traits.h>
#include <flare/core/common/error.h>
#include <flare/core/policy/analyze_policy.h>
#include <flare/core/common/concepts.h>
#include <typeinfo>

//----------------------------------------------------------------------------

namespace flare {

struct ParallelForTag {};
struct ParallelScanTag {};
struct ParallelReduceTag {};

struct ChunkSize {
  int value;
  ChunkSize(int value_) : value(value_) {}
};

/** \brief  Execution policy for work over a range of an integral type.
 *
 * Valid template argument options:
 *
 *  With a specified execution space:
 *    < ExecSpace , WorkTag , { IntConst | IntType } >
 *    < ExecSpace , WorkTag , void >
 *    < ExecSpace , { IntConst | IntType } , void >
 *    < ExecSpace , void , void >
 *
 *  With the default execution space:
 *    < WorkTag , { IntConst | IntType } , void >
 *    < WorkTag , void , void >
 *    < { IntConst | IntType } , void , void >
 *    < void , void , void >
 *
 *  IntType  is a fundamental integral type
 *  IntConst is an detail::integral_constant< IntType , Blocking >
 *
 *  Blocking is the granularity of partitioning the range among threads.
 */
template <class... Properties>
class RangePolicy : public detail::PolicyTraits<Properties...> {
 public:
  using traits = detail::PolicyTraits<Properties...>;

 private:
  typename traits::execution_space m_space;
  typename traits::index_type m_begin;
  typename traits::index_type m_end;
  typename traits::index_type m_granularity;
  typename traits::index_type m_granularity_mask;

  template <class... OtherProperties>
  friend class RangePolicy;

 public:
  //! Tag this class as an execution policy
  using execution_policy = RangePolicy<Properties...>;
  using member_type      = typename traits::index_type;
  using index_type       = typename traits::index_type;

  FLARE_INLINE_FUNCTION const typename traits::execution_space& space() const {
    return m_space;
  }
  FLARE_INLINE_FUNCTION member_type begin() const { return m_begin; }
  FLARE_INLINE_FUNCTION member_type end() const { return m_end; }

  // TODO: find a better workaround for Clangs weird instantiation order
  // This thing is here because of an instantiation error, where the RangePolicy
  // is inserted into FunctorValue Traits, which tries decltype on the operator.
  // It tries to do this even though the first argument of parallel for clearly
  // doesn't match.
  void operator()(const int&) const {}

  template <class... OtherProperties>
  RangePolicy(const RangePolicy<OtherProperties...>& p)
      : traits(p),  // base class may contain data such as desired occupancy
        m_space(p.m_space),
        m_begin(p.m_begin),
        m_end(p.m_end),
        m_granularity(p.m_granularity),
        m_granularity_mask(p.m_granularity_mask) {}

  inline RangePolicy()
      : m_space(),
        m_begin(0),
        m_end(0),
        m_granularity(0),
        m_granularity_mask(0) {}

  /** \brief  Total range */
  inline RangePolicy(const typename traits::execution_space& work_space,
                     const member_type work_begin, const member_type work_end)
      : m_space(work_space),
        m_begin(work_begin < work_end ? work_begin : 0),
        m_end(work_begin < work_end ? work_end : 0),
        m_granularity(0),
        m_granularity_mask(0) {
    set_auto_chunk_size();
  }

  /** \brief  Total range */
  inline RangePolicy(const member_type work_begin, const member_type work_end)
      : RangePolicy(typename traits::execution_space(), work_begin, work_end) {
    set_auto_chunk_size();
  }

  /** \brief  Total range */
  template <class... Args>
  inline RangePolicy(const typename traits::execution_space& work_space,
                     const member_type work_begin, const member_type work_end,
                     Args... args)
      : m_space(work_space),
        m_begin(work_begin < work_end ? work_begin : 0),
        m_end(work_begin < work_end ? work_end : 0),
        m_granularity(0),
        m_granularity_mask(0) {
    set_auto_chunk_size();
    set(args...);
  }

  /** \brief  Total range */
  template <class... Args>
  inline RangePolicy(const member_type work_begin, const member_type work_end,
                     Args... args)
      : RangePolicy(typename traits::execution_space(), work_begin, work_end) {
    set_auto_chunk_size();
    set(args...);
  }

 private:
  inline void set() {}

 public:
  template <class... Args>
  inline void set(Args...) {
    static_assert(
        0 == sizeof...(Args),
        "flare::RangePolicy: unhandled constructor arguments encountered.");
  }

  template <class... Args>
  inline void set(const ChunkSize& chunksize, Args... args) {
    m_granularity      = chunksize.value;
    m_granularity_mask = m_granularity - 1;
    set(args...);
  }

 public:
  /** \brief return chunk_size */
  inline member_type chunk_size() const { return m_granularity; }

  /** \brief set chunk_size to a discrete value*/
  inline RangePolicy& set_chunk_size(int chunk_size) {
    m_granularity      = chunk_size;
    m_granularity_mask = m_granularity - 1;
    return *this;
  }

 private:
  /** \brief finalize chunk_size if it was set to AUTO*/
  inline void set_auto_chunk_size() {
    auto concurrency = static_cast<int64_t>(m_space.concurrency());
    if (concurrency == 0) concurrency = 1;

    if (m_granularity > 0) {
      if (!detail::is_integral_power_of_two(m_granularity))
        flare::abort("RangePolicy blocking granularity must be power of two");
    }

    int64_t new_chunk_size = 1;
    while (new_chunk_size * 100 * concurrency <
           static_cast<int64_t>(m_end - m_begin))
      new_chunk_size *= 2;
    if (new_chunk_size < 128) {
      new_chunk_size = 1;
      while ((new_chunk_size * 40 * concurrency <
              static_cast<int64_t>(m_end - m_begin)) &&
             (new_chunk_size < 128))
        new_chunk_size *= 2;
    }
    m_granularity      = new_chunk_size;
    m_granularity_mask = m_granularity - 1;
  }

 public:
  /** \brief  Subrange for a partition's rank and size.
   *
   *  Typically used to partition a range over a group of threads.
   */
  struct WorkRange {
    using work_tag    = typename RangePolicy<Properties...>::work_tag;
    using member_type = typename RangePolicy<Properties...>::member_type;

    FLARE_INLINE_FUNCTION member_type begin() const { return m_begin; }
    FLARE_INLINE_FUNCTION member_type end() const { return m_end; }

    /** \brief  Subrange for a partition's rank and size.
     *
     *  Typically used to partition a range over a group of threads.
     */
    FLARE_INLINE_FUNCTION
    WorkRange(const RangePolicy& range, const int part_rank,
              const int part_size)
        : m_begin(0), m_end(0) {
      if (part_size) {
        // Split evenly among partitions, then round up to the granularity.
        const member_type work_part =
            ((((range.end() - range.begin()) + (part_size - 1)) / part_size) +
             range.m_granularity_mask) &
            ~member_type(range.m_granularity_mask);

        m_begin = range.begin() + work_part * part_rank;
        m_end   = m_begin + work_part;

        if (range.end() < m_begin) m_begin = range.end();
        if (range.end() < m_end) m_end = range.end();
      }
    }

   private:
    member_type m_begin;
    member_type m_end;
    WorkRange();
    WorkRange& operator=(const WorkRange&);
  };
};

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

namespace detail {

template <class ExecSpace, class... Properties>
class TeamPolicyInternal : public detail::PolicyTraits<Properties...> {
 private:
  using traits = detail::PolicyTraits<Properties...>;

 public:
  using index_type = typename traits::index_type;

  //----------------------------------------
  /** \brief  Query maximum team size for a given functor.
   *
   *  This size takes into account execution space concurrency limitations and
   *  scratch memory space limitations for reductions, team reduce/scan, and
   *  team shared memory.
   *
   *  This function only works for single-operator functors.
   *  With multi-operator functors it cannot be determined
   *  which operator will be called.
   */
  template <class FunctorType>
  static int team_size_max(const FunctorType&);

  /** \brief  Query recommended team size for a given functor.
   *
   *  This size takes into account execution space concurrency limitations and
   *  scratch memory space limitations for reductions, team reduce/scan, and
   *  team shared memory.
   *
   *  This function only works for single-operator functors.
   *  With multi-operator functors it cannot be determined
   *  which operator will be called.
   */
  template <class FunctorType>
  static int team_size_recommended(const FunctorType&);

  template <class FunctorType>
  static int team_size_recommended(const FunctorType&, const int&);

  template <class FunctorType>
  int team_size_recommended(const FunctorType& functor,
                            const int vector_length);

  //----------------------------------------
  /** \brief  Construct policy with the given instance of the execution space */
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, int team_size_request,
                     int vector_length_request = 1);

  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, const flare::AUTO_t&,
                     int vector_length_request = 1);

  /** \brief  Construct policy with the default instance of the execution space
   */
  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int vector_length_request = 1);

  TeamPolicyInternal(int league_size_request, const flare::AUTO_t&,
                     int vector_length_request = 1);

  /*  TeamPolicyInternal( int league_size_request , int team_size_request );

    TeamPolicyInternal( int league_size_request , const flare::AUTO_t & );*/

  /** \brief  The actual league size (number of teams) of the policy.
   *
   *  This may be smaller than the requested league size due to limitations
   *  of the execution space.
   */
  FLARE_INLINE_FUNCTION int league_size() const;

  /** \brief  The actual team size (number of threads per team) of the policy.
   *
   *  This may be smaller than the requested team size due to limitations
   *  of the execution space.
   */
  FLARE_INLINE_FUNCTION int team_size() const;

  /** \brief Whether the policy has an automatically determined team size
   */
  inline bool impl_auto_team_size() const;
  /** \brief Whether the policy has an automatically determined vector length
   */
  inline bool impl_auto_vector_length() const;

  static int vector_length_max();

  FLARE_INLINE_FUNCTION int impl_vector_length() const;

  inline typename traits::index_type chunk_size() const;

  inline TeamPolicyInternal& set_chunk_size(int chunk_size);

  /** \brief  Parallel execution of a functor calls the functor once with
   *          each member of the execution policy.
   */
  struct member_type {
    /** \brief  Handle to the currently executing team shared scratch memory */
    FLARE_INLINE_FUNCTION
    typename traits::execution_space::scratch_memory_space team_shmem() const;

    /** \brief  Rank of this team within the league of teams */
    FLARE_INLINE_FUNCTION int league_rank() const;

    /** \brief  Number of teams in the league */
    FLARE_INLINE_FUNCTION int league_size() const;

    /** \brief  Rank of this thread within this team */
    FLARE_INLINE_FUNCTION int team_rank() const;

    /** \brief  Number of threads in this team */
    FLARE_INLINE_FUNCTION int team_size() const;

    /** \brief  Barrier among the threads of this team */
    FLARE_INLINE_FUNCTION void team_barrier() const;

    /** \brief  Intra-team reduction. Returns join of all values of the team
     * members. */
    template <class JoinOp>
    FLARE_INLINE_FUNCTION typename JoinOp::value_type team_reduce(
        const typename JoinOp::value_type, const JoinOp&) const;

    /** \brief  Intra-team exclusive prefix sum with team_rank() ordering.
     *
     *  The highest rank thread can compute the reduction total as
     *    reduction_total = dev.team_scan( value ) + value ;
     */
    template <typename Type>
    FLARE_INLINE_FUNCTION Type team_scan(const Type& value) const;

    /** \brief  Intra-team exclusive prefix sum with team_rank() ordering
     *          with intra-team non-deterministic ordering accumulation.
     *
     *  The global inter-team accumulation value will, at the end of the
     *  league's parallel execution, be the scan's total.
     *  Parallel execution ordering of the league's teams is non-deterministic.
     *  As such the base value for each team's scan operation is similarly
     *  non-deterministic.
     */
    template <typename Type>
    FLARE_INLINE_FUNCTION Type team_scan(const Type& value,
                                          Type* const global_accum) const;
  };
};

struct PerTeamValue {
  size_t value;
  PerTeamValue(size_t arg);
};

struct PerThreadValue {
  size_t value;
  PerThreadValue(size_t arg);
};

template <class iType, class... Args>
struct ExtractVectorLength {
  static inline iType value(
      std::enable_if_t<std::is_integral<iType>::value, iType> val, Args...) {
    return val;
  }
  static inline std::enable_if_t<!std::is_integral<iType>::value, int> value(
      std::enable_if_t<!std::is_integral<iType>::value, iType>, Args...) {
    return 1;
  }
};

template <class iType, class... Args>
inline std::enable_if_t<std::is_integral<iType>::value, iType>
extract_vector_length(iType val, Args...) {
  return val;
}

template <class iType, class... Args>
inline std::enable_if_t<!std::is_integral<iType>::value, int>
extract_vector_length(iType, Args...) {
  return 1;
}

}  // namespace detail

detail::PerTeamValue PerTeam(const size_t& arg);
detail::PerThreadValue PerThread(const size_t& arg);

struct ScratchRequest {
  int level;

  size_t per_team;
  size_t per_thread;

  inline ScratchRequest(const int& level_,
                        const detail::PerTeamValue& team_value) {
    level      = level_;
    per_team   = team_value.value;
    per_thread = 0;
  }

  inline ScratchRequest(const int& level_,
                        const detail::PerThreadValue& thread_value) {
    level      = level_;
    per_team   = 0;
    per_thread = thread_value.value;
  }

  inline ScratchRequest(const int& level_, const detail::PerTeamValue& team_value,
                        const detail::PerThreadValue& thread_value) {
    level      = level_;
    per_team   = team_value.value;
    per_thread = thread_value.value;
  }

  inline ScratchRequest(const int& level_,
                        const detail::PerThreadValue& thread_value,
                        const detail::PerTeamValue& team_value) {
    level      = level_;
    per_team   = team_value.value;
    per_thread = thread_value.value;
  }
};

// Throws a runtime exception if level is not `0` or `1`
void team_policy_check_valid_storage_level_argument(int level);

/** \brief  Execution policy for parallel work over a league of teams of
 * threads.
 *
 *  The work functor is called for each thread of each team such that
 *  the team's member threads are guaranteed to be concurrent.
 *
 *  The team's threads have access to team shared scratch memory and
 *  team collective operations.
 *
 *  If the WorkTag is non-void then the first calling argument of the
 *  work functor's parentheses operator is 'const WorkTag &'.
 *  This allows a functor to have multiple work member functions.
 *
 *  Order of template arguments does not matter, since the implementation
 *  uses variadic templates. Each and any of the template arguments can
 *  be omitted.
 *
 *  Possible Template arguments and their default values:
 *    ExecutionSpace (DefaultExecutionSpace): where to execute code. Must be
 * enabled. WorkTag (none): Tag which is used as the first argument for the
 * functor operator. Schedule<Type> (Schedule<Static>): Scheduling Policy
 * (Dynamic, or Static). IndexType<Type> (IndexType<ExecutionSpace::size_type>:
 * Integer Index type used to iterate over the Index space.
 *    LaunchBounds<unsigned,unsigned> Launch Bounds for CUDA compilation,
 *    default of LaunchBounds<0,0> indicates no launch bounds specified.
 */
template <class... Properties>
class TeamPolicy
    : public detail::TeamPolicyInternal<
          typename detail::PolicyTraits<Properties...>::execution_space,
          Properties...> {
  using internal_policy = detail::TeamPolicyInternal<
      typename detail::PolicyTraits<Properties...>::execution_space,
      Properties...>;

  template <class... OtherProperties>
  friend class TeamPolicy;

 public:
  using traits = detail::PolicyTraits<Properties...>;

  using execution_policy = TeamPolicy<Properties...>;

  TeamPolicy() : internal_policy(0, AUTO) {}

  /** \brief  Construct policy with the given instance of the execution space */
  TeamPolicy(const typename traits::execution_space& space_,
             int league_size_request, int team_size_request,
             int vector_length_request = 1)
      : internal_policy(space_, league_size_request, team_size_request,
                        vector_length_request) {}

  TeamPolicy(const typename traits::execution_space& space_,
             int league_size_request, const flare::AUTO_t&,
             int vector_length_request = 1)
      : internal_policy(space_, league_size_request, flare::AUTO(),
                        vector_length_request) {}

  TeamPolicy(const typename traits::execution_space& space_,
             int league_size_request, const flare::AUTO_t&,
             const flare::AUTO_t&)
      : internal_policy(space_, league_size_request, flare::AUTO(),
                        flare::AUTO()) {}
  TeamPolicy(const typename traits::execution_space& space_,
             int league_size_request, const int team_size_request,
             const flare::AUTO_t&)
      : internal_policy(space_, league_size_request, team_size_request,
                        flare::AUTO()) {}
  /** \brief  Construct policy with the default instance of the execution space
   */
  TeamPolicy(int league_size_request, int team_size_request,
             int vector_length_request = 1)
      : internal_policy(league_size_request, team_size_request,
                        vector_length_request) {}

  TeamPolicy(int league_size_request, const flare::AUTO_t&,
             int vector_length_request = 1)
      : internal_policy(league_size_request, flare::AUTO(),
                        vector_length_request) {}

  TeamPolicy(int league_size_request, const flare::AUTO_t&,
             const flare::AUTO_t&)
      : internal_policy(league_size_request, flare::AUTO(), flare::AUTO()) {}
  TeamPolicy(int league_size_request, const int team_size_request,
             const flare::AUTO_t&)
      : internal_policy(league_size_request, team_size_request,
                        flare::AUTO()) {}

  template <class... OtherProperties>
  TeamPolicy(const TeamPolicy<OtherProperties...> p) : internal_policy(p) {
    // Cannot call converting constructor in the member initializer list because
    // it is not a direct base.
    internal_policy::traits::operator=(p);
  }

 private:
  TeamPolicy(const internal_policy& p) : internal_policy(p) {}

 public:
  inline TeamPolicy& set_chunk_size(int chunk) {
    static_assert(std::is_same<decltype(internal_policy::set_chunk_size(chunk)),
                               internal_policy&>::value,
                  "internal set_chunk_size should return a reference");
    return static_cast<TeamPolicy&>(internal_policy::set_chunk_size(chunk));
  }

  inline TeamPolicy& set_scratch_size(const int& level,
                                      const detail::PerTeamValue& per_team) {
    static_assert(std::is_same<decltype(internal_policy::set_scratch_size(
                                   level, per_team)),
                               internal_policy&>::value,
                  "internal set_chunk_size should return a reference");

    team_policy_check_valid_storage_level_argument(level);
    return static_cast<TeamPolicy&>(
        internal_policy::set_scratch_size(level, per_team));
  }
  inline TeamPolicy& set_scratch_size(const int& level,
                                      const detail::PerThreadValue& per_thread) {
    team_policy_check_valid_storage_level_argument(level);
    return static_cast<TeamPolicy&>(
        internal_policy::set_scratch_size(level, per_thread));
  }
  inline TeamPolicy& set_scratch_size(const int& level,
                                      const detail::PerTeamValue& per_team,
                                      const detail::PerThreadValue& per_thread) {
    team_policy_check_valid_storage_level_argument(level);
    return static_cast<TeamPolicy&>(
        internal_policy::set_scratch_size(level, per_team, per_thread));
  }
  inline TeamPolicy& set_scratch_size(const int& level,
                                      const detail::PerThreadValue& per_thread,
                                      const detail::PerTeamValue& per_team) {
    team_policy_check_valid_storage_level_argument(level);
    return static_cast<TeamPolicy&>(
        internal_policy::set_scratch_size(level, per_team, per_thread));
  }
};

namespace detail {

template <typename iType, class TeamMemberType>
struct TeamThreadRangeBoundariesStruct {
 private:
  FLARE_INLINE_FUNCTION static iType ibegin(const iType& arg_begin,
                                             const iType& arg_end,
                                             const iType& arg_rank,
                                             const iType& arg_size) {
    return arg_begin +
           ((arg_end - arg_begin + arg_size - 1) / arg_size) * arg_rank;
  }

  FLARE_INLINE_FUNCTION static iType iend(const iType& arg_begin,
                                           const iType& arg_end,
                                           const iType& arg_rank,
                                           const iType& arg_size) {
    const iType end_ =
        arg_begin +
        ((arg_end - arg_begin + arg_size - 1) / arg_size) * (arg_rank + 1);
    return end_ < arg_end ? end_ : arg_end;
  }

 public:
  using index_type = iType;
  const iType start;
  const iType end;
  enum { increment = 1 };
  const TeamMemberType& thread;

  FLARE_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct(const TeamMemberType& arg_thread,
                                  const iType& arg_end)
      : start(
            ibegin(0, arg_end, arg_thread.team_rank(), arg_thread.team_size())),
        end(iend(0, arg_end, arg_thread.team_rank(), arg_thread.team_size())),
        thread(arg_thread) {}

  FLARE_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct(const TeamMemberType& arg_thread,
                                  const iType& arg_begin, const iType& arg_end)
      : start(ibegin(arg_begin, arg_end, arg_thread.team_rank(),
                     arg_thread.team_size())),
        end(iend(arg_begin, arg_end, arg_thread.team_rank(),
                 arg_thread.team_size())),
        thread(arg_thread) {}
};

template <typename iType, class TeamMemberType>
struct TeamVectorRangeBoundariesStruct {
 private:
  FLARE_INLINE_FUNCTION static iType ibegin(const iType& arg_begin,
                                             const iType& arg_end,
                                             const iType& arg_rank,
                                             const iType& arg_size) {
    return arg_begin +
           ((arg_end - arg_begin + arg_size - 1) / arg_size) * arg_rank;
  }

  FLARE_INLINE_FUNCTION static iType iend(const iType& arg_begin,
                                           const iType& arg_end,
                                           const iType& arg_rank,
                                           const iType& arg_size) {
    const iType end_ =
        arg_begin +
        ((arg_end - arg_begin + arg_size - 1) / arg_size) * (arg_rank + 1);
    return end_ < arg_end ? end_ : arg_end;
  }

 public:
  using index_type = iType;
  const iType start;
  const iType end;
  enum { increment = 1 };
  const TeamMemberType& thread;

  FLARE_INLINE_FUNCTION
  TeamVectorRangeBoundariesStruct(const TeamMemberType& arg_thread,
                                  const iType& arg_end)
      : start(
            ibegin(0, arg_end, arg_thread.team_rank(), arg_thread.team_size())),
        end(iend(0, arg_end, arg_thread.team_rank(), arg_thread.team_size())),
        thread(arg_thread) {}

  FLARE_INLINE_FUNCTION
  TeamVectorRangeBoundariesStruct(const TeamMemberType& arg_thread,
                                  const iType& arg_begin, const iType& arg_end)
      : start(ibegin(arg_begin, arg_end, arg_thread.team_rank(),
                     arg_thread.team_size())),
        end(iend(arg_begin, arg_end, arg_thread.team_rank(),
                 arg_thread.team_size())),
        thread(arg_thread) {}
};

template <typename iType, class TeamMemberType>
struct ThreadVectorRangeBoundariesStruct {
  using index_type = iType;
  const index_type start;
  const index_type end;
  enum { increment = 1 };

  FLARE_INLINE_FUNCTION
  constexpr ThreadVectorRangeBoundariesStruct(const TeamMemberType,
                                              const index_type& count) noexcept
      : start(static_cast<index_type>(0)), end(count) {}

  FLARE_INLINE_FUNCTION
  constexpr ThreadVectorRangeBoundariesStruct(
      const TeamMemberType, const index_type& arg_begin,
      const index_type& arg_end) noexcept
      : start(static_cast<index_type>(arg_begin)), end(arg_end) {}
};

template <class TeamMemberType>
struct ThreadSingleStruct {
  const TeamMemberType& team_member;
  FLARE_INLINE_FUNCTION
  ThreadSingleStruct(const TeamMemberType& team_member_)
      : team_member(team_member_) {}
};

template <class TeamMemberType>
struct VectorSingleStruct {
  const TeamMemberType& team_member;
  FLARE_INLINE_FUNCTION
  VectorSingleStruct(const TeamMemberType& team_member_)
      : team_member(team_member_) {}
};

}  // namespace detail

/** \brief  Execution policy for parallel work over a threads within a team.
 *
 *  The range is split over all threads in a team. The Mapping scheme depends on
 * the architecture. This policy is used together with a parallel pattern as a
 * nested layer within a kernel launched with the TeamPolicy. This variant
 * expects a single count. So the range is (0,count].
 */
template <typename iType, class TeamMemberType, class _never_use_this_overload>
FLARE_INLINE_FUNCTION_DELETED
    detail::TeamThreadRangeBoundariesStruct<iType, TeamMemberType>
    TeamThreadRange(const TeamMemberType&, const iType& count) = delete;

/** \brief  Execution policy for parallel work over a threads within a team.
 *
 *  The range is split over all threads in a team. The Mapping scheme depends on
 * the architecture. This policy is used together with a parallel pattern as a
 * nested layer within a kernel launched with the TeamPolicy. This variant
 * expects a begin and end. So the range is (begin,end].
 */
template <typename iType1, typename iType2, class TeamMemberType,
          class _never_use_this_overload>
FLARE_INLINE_FUNCTION_DELETED detail::TeamThreadRangeBoundariesStruct<
    std::common_type_t<iType1, iType2>, TeamMemberType>
TeamThreadRange(const TeamMemberType&, const iType1& begin,
                const iType2& end) = delete;

/** \brief  Execution policy for parallel work over a threads within a team.
 *
 *  The range is split over all threads in a team. The Mapping scheme depends on
 * the architecture. This policy is used together with a parallel pattern as a
 * nested layer within a kernel launched with the TeamPolicy. This variant
 * expects a single count. So the range is (0,count].
 */
template <typename iType, class TeamMemberType, class _never_use_this_overload>
FLARE_INLINE_FUNCTION_DELETED
    detail::TeamThreadRangeBoundariesStruct<iType, TeamMemberType>
    TeamVectorRange(const TeamMemberType&, const iType& count) = delete;

/** \brief  Execution policy for parallel work over a threads within a team.
 *
 *  The range is split over all threads in a team. The Mapping scheme depends on
 * the architecture. This policy is used together with a parallel pattern as a
 * nested layer within a kernel launched with the TeamPolicy. This variant
 * expects a begin and end. So the range is (begin,end].
 */
template <typename iType1, typename iType2, class TeamMemberType,
          class _never_use_this_overload>
FLARE_INLINE_FUNCTION_DELETED detail::TeamThreadRangeBoundariesStruct<
    std::common_type_t<iType1, iType2>, TeamMemberType>
TeamVectorRange(const TeamMemberType&, const iType1& begin,
                const iType2& end) = delete;

/** \brief  Execution policy for a vector parallel loop.
 *
 *  The range is split over all vector lanes in a thread. The Mapping scheme
 * depends on the architecture. This policy is used together with a parallel
 * pattern as a nested layer within a kernel launched with the TeamPolicy. This
 * variant expects a single count. So the range is (0,count].
 */
template <typename iType, class TeamMemberType, class _never_use_this_overload>
FLARE_INLINE_FUNCTION_DELETED
    detail::ThreadVectorRangeBoundariesStruct<iType, TeamMemberType>
    ThreadVectorRange(const TeamMemberType&, const iType& count) = delete;

template <typename iType1, typename iType2, class TeamMemberType,
          class _never_use_this_overload>
FLARE_INLINE_FUNCTION_DELETED detail::ThreadVectorRangeBoundariesStruct<
    std::common_type_t<iType1, iType2>, TeamMemberType>
ThreadVectorRange(const TeamMemberType&, const iType1& arg_begin,
                  const iType2& arg_end) = delete;

namespace detail {

enum class TeamMDRangeLastNestLevel : bool { NotLastNestLevel, LastNestLevel };
enum class TeamMDRangeParThread : bool { NotParThread, ParThread };
enum class TeamMDRangeParVector : bool { NotParVector, ParVector };
enum class TeamMDRangeThreadAndVector : bool { NotBoth, Both };

template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct HostBasedNestLevel;

template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct AcceleratorBasedNestLevel;

// ThreadAndVectorNestLevel determines on which nested level parallelization
// happens.
//   - Rank is flare::Rank<TotalNestLevel, Iter>
//     - TotalNestLevel is the total number of loop nests
//     - Iter is whether to go forward or backward through ranks (i.e. the
//       iteration order for MDRangePolicy)
//   - ThreadAndVector determines whether both vector and thread parallelism is
//     in use
template <typename Rank, typename ExecSpace,
          TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel;

struct NoReductionTag {};

template <typename Rank, typename TeamMDPolicy, typename Lambda,
          typename ReductionValueType>
FLARE_INLINE_FUNCTION void md_parallel_impl(TeamMDPolicy const& policy,
                                             Lambda const& lambda,
                                             ReductionValueType&& val);
}  // namespace detail

template <typename Rank, typename TeamHandle>
struct TeamThreadMDRange;

template <unsigned N, Iterate OuterDir, Iterate InnerDir, typename TeamHandle>
struct TeamThreadMDRange<Rank<N, OuterDir, InnerDir>, TeamHandle> {
  using NestLevelType  = int;
  using BoundaryType   = int;
  using TeamHandleType = TeamHandle;
  using ExecutionSpace = typename TeamHandleType::execution_space;
  using ArrayLayout    = typename ExecutionSpace::array_layout;

  static constexpr NestLevelType total_nest_level =
      Rank<N, OuterDir, InnerDir>::rank;
  static constexpr Iterate iter    = OuterDir;
  static constexpr auto par_thread = detail::TeamMDRangeParThread::ParThread;
  static constexpr auto par_vector = detail::TeamMDRangeParVector::NotParVector;

  static constexpr Iterate direction =
      OuterDir == Iterate::Default
          ? layout_iterate_type_selector<ArrayLayout>::outer_iteration_pattern
          : iter;

  template <class... Args>
  FLARE_FUNCTION TeamThreadMDRange(TeamHandleType const& team_, Args&&... args)
      : team(team_), boundaries{static_cast<BoundaryType>(args)...} {
    static_assert(sizeof...(Args) == total_nest_level);
  }

  TeamHandleType const& team;
  BoundaryType boundaries[total_nest_level];
};

template <typename TeamHandle, typename... Args>
TeamThreadMDRange(TeamHandle const&, Args&&...)
    ->TeamThreadMDRange<Rank<sizeof...(Args), Iterate::Default>, TeamHandle>;

template <typename Rank, typename TeamHandle>
struct ThreadVectorMDRange;

template <unsigned N, Iterate OuterDir, Iterate InnerDir, typename TeamHandle>
struct ThreadVectorMDRange<Rank<N, OuterDir, InnerDir>, TeamHandle> {
  using NestLevelType  = int;
  using BoundaryType   = int;
  using TeamHandleType = TeamHandle;
  using ExecutionSpace = typename TeamHandleType::execution_space;
  using ArrayLayout    = typename ExecutionSpace::array_layout;

  static constexpr NestLevelType total_nest_level =
      Rank<N, OuterDir, InnerDir>::rank;
  static constexpr Iterate iter    = OuterDir;
  static constexpr auto par_thread = detail::TeamMDRangeParThread::NotParThread;
  static constexpr auto par_vector = detail::TeamMDRangeParVector::ParVector;

  static constexpr Iterate direction =
      OuterDir == Iterate::Default
          ? layout_iterate_type_selector<ArrayLayout>::outer_iteration_pattern
          : iter;

  template <class... Args>
  FLARE_INLINE_FUNCTION ThreadVectorMDRange(TeamHandleType const& team_,
                                             Args&&... args)
      : team(team_), boundaries{static_cast<BoundaryType>(args)...} {
    static_assert(sizeof...(Args) == total_nest_level);
  }

  TeamHandleType const& team;
  BoundaryType boundaries[total_nest_level];
};

template <typename TeamHandle, typename... Args>
ThreadVectorMDRange(TeamHandle const&, Args&&...)
    ->ThreadVectorMDRange<Rank<sizeof...(Args), Iterate::Default>, TeamHandle>;

template <typename Rank, typename TeamHandle>
struct TeamVectorMDRange;

template <unsigned N, Iterate OuterDir, Iterate InnerDir, typename TeamHandle>
struct TeamVectorMDRange<Rank<N, OuterDir, InnerDir>, TeamHandle> {
  using NestLevelType  = int;
  using BoundaryType   = int;
  using TeamHandleType = TeamHandle;
  using ExecutionSpace = typename TeamHandleType::execution_space;
  using ArrayLayout    = typename ExecutionSpace::array_layout;

  static constexpr NestLevelType total_nest_level =
      Rank<N, OuterDir, InnerDir>::rank;
  static constexpr Iterate iter    = OuterDir;
  static constexpr auto par_thread = detail::TeamMDRangeParThread::ParThread;
  static constexpr auto par_vector = detail::TeamMDRangeParVector::ParVector;

  static constexpr Iterate direction =
      iter == Iterate::Default
          ? layout_iterate_type_selector<ArrayLayout>::outer_iteration_pattern
          : iter;

  template <class... Args>
  FLARE_INLINE_FUNCTION TeamVectorMDRange(TeamHandleType const& team_,
                                           Args&&... args)
      : team(team_), boundaries{static_cast<BoundaryType>(args)...} {
    static_assert(sizeof...(Args) == total_nest_level);
  }

  TeamHandleType const& team;
  BoundaryType boundaries[total_nest_level];
};

template <typename TeamHandle, typename... Args>
TeamVectorMDRange(TeamHandle const&, Args&&...)
    ->TeamVectorMDRange<Rank<sizeof...(Args), Iterate::Default>, TeamHandle>;

template <typename Rank, typename TeamHandle, typename Lambda,
          typename ReducerValueType>
FLARE_INLINE_FUNCTION void parallel_reduce(
    TeamThreadMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda,
    ReducerValueType& val) {
  detail::md_parallel_impl<Rank>(policy, lambda, val);
}

template <typename Rank, typename TeamHandle, typename Lambda>
FLARE_INLINE_FUNCTION void parallel_for(
    TeamThreadMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda) {
  detail::md_parallel_impl<Rank>(policy, lambda, detail::NoReductionTag());
}

template <typename Rank, typename TeamHandle, typename Lambda,
          typename ReducerValueType>
FLARE_INLINE_FUNCTION void parallel_reduce(
    ThreadVectorMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda,
    ReducerValueType& val) {
  detail::md_parallel_impl<Rank>(policy, lambda, val);
}

template <typename Rank, typename TeamHandle, typename Lambda>
FLARE_INLINE_FUNCTION void parallel_for(
    ThreadVectorMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda) {
  detail::md_parallel_impl<Rank>(policy, lambda, detail::NoReductionTag());
}

template <typename Rank, typename TeamHandle, typename Lambda,
          typename ReducerValueType>
FLARE_INLINE_FUNCTION void parallel_reduce(
    TeamVectorMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda,
    ReducerValueType& val) {
  detail::md_parallel_impl<Rank>(policy, lambda, val);
}

template <typename Rank, typename TeamHandle, typename Lambda>
FLARE_INLINE_FUNCTION void parallel_for(
    TeamVectorMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda) {
  detail::md_parallel_impl<Rank>(policy, lambda, detail::NoReductionTag());
}

namespace detail {

template <typename FunctorType, typename TagType,
          bool HasTag = !std::is_void<TagType>::value>
struct ParallelConstructName;

template <typename FunctorType, typename TagType>
struct ParallelConstructName<FunctorType, TagType, true> {
  ParallelConstructName(std::string const& label) : label_ref(label) {
    if (label.empty()) {
      default_name = std::string(typeid(FunctorType).name()) + "/" +
                     typeid(TagType).name();
    }
  }
  std::string const& get() {
    return (label_ref.empty()) ? default_name : label_ref;
  }
  std::string const& label_ref;
  std::string default_name;
};

template <typename FunctorType, typename TagType>
struct ParallelConstructName<FunctorType, TagType, false> {
  ParallelConstructName(std::string const& label) : label_ref(label) {
    if (label.empty()) {
      default_name = std::string(typeid(FunctorType).name());
    }
  }
  std::string const& get() {
    return (label_ref.empty()) ? default_name : label_ref;
  }
  std::string const& label_ref;
  std::string default_name;
};

}  // namespace detail

}  // namespace flare

namespace flare {

namespace detail {

template <class PatternTag, class... Args>
struct PatternImplSpecializationFromTag;

template <class... Args>
struct PatternImplSpecializationFromTag<flare::ParallelForTag, Args...>
    : type_identity<ParallelFor<Args...>> {};

template <class... Args>
struct PatternImplSpecializationFromTag<flare::ParallelReduceTag, Args...>
    : type_identity<ParallelReduce<Args...>> {};

template <class... Args>
struct PatternImplSpecializationFromTag<flare::ParallelScanTag, Args...>
    : type_identity<ParallelScan<Args...>> {};

template <class PatternImpl>
struct PatternTagFromImplSpecialization;

template <class... Args>
struct PatternTagFromImplSpecialization<ParallelFor<Args...>>
    : type_identity<ParallelForTag> {};

template <class... Args>
struct PatternTagFromImplSpecialization<ParallelReduce<Args...>>
    : type_identity<ParallelReduceTag> {};

template <class... Args>
struct PatternTagFromImplSpecialization<ParallelScan<Args...>>
    : type_identity<ParallelScanTag> {};

}  // end namespace detail

}  // namespace flare
#endif  // FLARE_CORE_POLICY_EXEC_POLICY_H_
