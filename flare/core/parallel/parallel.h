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

/// \file parallel.h
/// \brief Declaration of parallel operators


#ifndef FLARE_CORE_PARALLEL_PARALLEL_H_
#define FLARE_CORE_PARALLEL_PARALLEL_H_

#include <flare/core_fwd.h>
#include <flare/core/common/detection_idiom.h>
#include <flare/core/policy/exec_policy.h>
#include <flare/core/tensor/tensor.h>

#include <flare/core/profile/tools.h>
#include <flare/core/profile/tools_generic.h>

#include <flare/core/common/traits.h>
#include <flare/core/common/functor_analysis.h>

#include <cstddef>
#include <type_traits>
#include <typeinfo>

namespace flare::detail {

    template<class T>
    using execution_space_t = typename T::execution_space;

    template<class T>
    using device_type_t = typename T::device_type;

//----------------------------------------------------------------------------
/** \brief  Given a Functor and Execution Policy query an execution space.
 *
 *  if       the Policy has an execution space use that
 *  else if  the Functor has an execution_space use that
 *  else if  the Functor has a device_type use that for backward compatibility
 *  else     use the default
 */

    template<class Functor, class Policy>
    struct FunctorPolicyExecutionSpace {
        using policy_execution_space = detected_t<execution_space_t, Policy>;
        using functor_execution_space = detected_t<execution_space_t, Functor>;
        using functor_device_type = detected_t<device_type_t, Functor>;
        using functor_device_type_execution_space =
                detected_t<execution_space_t, functor_device_type>;

        static_assert(
                !is_detected<execution_space_t, Policy>::value ||
                !is_detected<execution_space_t, Functor>::value ||
                std::is_same<policy_execution_space, functor_execution_space>::value,
                "A policy with an execution space and a functor with an execution space "
                "are given but the execution space types do not match!");
        static_assert(!is_detected<execution_space_t, Policy>::value ||
                      !is_detected<device_type_t, Functor>::value ||
                      std::is_same<policy_execution_space,
                              functor_device_type_execution_space>::value,
                      "A policy with an execution space and a functor with a device "
                      "type are given but the execution space types do not match!");
        static_assert(!is_detected<device_type_t, Functor>::value ||
                      !is_detected<execution_space_t, Functor>::value ||
                      std::is_same<functor_device_type_execution_space,
                              functor_execution_space>::value,
                      "A functor with both an execution space and device type is "
                      "given but their execution space types do not match!");

        using execution_space = detected_or_t<
                detected_or_t<
                        std::conditional_t<
                                is_detected<device_type_t, Functor>::value,
                                detected_t<execution_space_t, detected_t<device_type_t, Functor>>,
                                flare::DefaultExecutionSpace>,
                        execution_space_t, Functor>,
                execution_space_t, Policy>;
    };

}  // namespace flare::detail

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

    /** \brief Execute \c functor in parallel according to the execution \c policy.
     *
     * A "functor" is a class containing the function to execute in parallel,
     * data needed for that execution, and an optional \c execution_space
     * alias.  Here is an example functor for parallel_for:
     *
     * \code
     *  class FunctorType {
     *  public:
     *    using execution_space = ...;
     *    void operator() ( WorkType iwork ) const ;
     *  };
     * \endcode
     *
     * In the above example, \c WorkType is any integer type for which a
     * valid conversion from \c size_t to \c IntType exists.  Its
     * <tt>operator()</tt> method defines the operation to parallelize,
     * over the range of integer indices <tt>iwork=[0,work_count-1]</tt>.
     * This compares to a single iteration \c iwork of a \c for loop.
     * If \c execution_space is not defined DefaultExecutionSpace will be used.
     */
    template<
            class ExecPolicy, class FunctorType,
            class Enable = std::enable_if_t<is_execution_policy<ExecPolicy>::value>>
    inline void parallel_for(const std::string &str, const ExecPolicy &policy,
                             const FunctorType &functor) {
        uint64_t kpID = 0;

        ExecPolicy inner_policy = policy;
        flare::Tools::detail::begin_parallel_for(inner_policy, functor, str, kpID);

        flare::detail::shared_allocation_tracking_disable();
        detail::ParallelFor<FunctorType, ExecPolicy> closure(functor, inner_policy);
        flare::detail::shared_allocation_tracking_enable();

        closure.execute();

        flare::Tools::detail::end_parallel_for(inner_policy, functor, str, kpID);
    }

    template<class ExecPolicy, class FunctorType>
    inline void parallel_for(
            const ExecPolicy &policy, const FunctorType &functor,
            std::enable_if_t<is_execution_policy<ExecPolicy>::value> * = nullptr) {
        flare::parallel_for("", policy, functor);
    }

    template<class FunctorType>
    inline void parallel_for(const std::string &str, const size_t work_count,
                             const FunctorType &functor) {
        using execution_space =
                typename detail::FunctorPolicyExecutionSpace<FunctorType,
                        void>::execution_space;
        using policy = RangePolicy<execution_space>;

        policy execution_policy = policy(0, work_count);
        ::flare::parallel_for(str, execution_policy, functor);
    }

    template<class FunctorType>
    inline void parallel_for(const size_t work_count, const FunctorType &functor) {
        ::flare::parallel_for("", work_count, functor);
    }

}  // namespace flare

#include <flare/core/parallel/parallel_reduce.h>
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

    /// \fn parallel_scan
    /// \tparam ExecutionPolicy The execution policy type.
    /// \tparam FunctorType     The scan functor type.
    ///
    /// \param policy  [in] The execution policy.
    /// \param functor [in] The scan functor.
    ///
    /// This function implements a parallel scan pattern.  The scan can
    /// be either inclusive or exclusive, depending on how you implement
    /// the scan functor.
    ///
    /// A scan functor looks almost exactly like a reduce functor, except
    /// that its operator() takes a third \c bool argument, \c final_pass,
    /// which indicates whether this is the last pass of the scan
    /// operation.  We will show below how to use the \c final_pass
    /// argument to control whether the scan is inclusive or exclusive.
    ///
    /// Here is the minimum required interface of a scan functor for a POD
    /// (plain old data) value type \c PodType.  That is, the result is a
    /// Tensor of zero or more PodType.  It is also possible for the result
    /// to be an array of (same-sized) arrays of PodType, but we do not
    /// show the required interface for that here.
    /// \code
    /// template< class ExecPolicy , class FunctorType >
    /// class ScanFunctor {
    /// public:
    ///   // The flare device type
    ///   using execution_space = ...;
    ///   // Type of an entry of the array containing the result;
    ///   // also the type of each of the entries combined using
    ///   // operator() or join().
    ///   using value_type = PodType;
    ///
    ///   void operator () (const ExecPolicy::member_type & i,
    ///                     value_type& update,
    ///                     const bool final_pass) const;
    ///   void init (value_type& update) const;
    ///   void join (value_type& update,
    //               const value_type& input) const
    /// };
    /// \endcode
    ///
    /// Here is an example of a functor which computes an inclusive plus-scan
    /// of an array of \c int, in place.  If given an array [1, 2, 3, 4], this
    /// scan will overwrite that array with [1, 3, 6, 10].
    ///
    /// \code
    /// template<class SpaceType>
    /// class InclScanFunctor {
    /// public:
    ///   using execution_space = SpaceType;
    ///   using value_type = int;
    ///   using size_type = typename SpaceType::size_type;
    ///
    ///   InclScanFunctor( flare::Tensor<value_type*, execution_space> x
    ///                  , flare::Tensor<value_type*, execution_space> y ) : m_x(x),
    ///                  m_y(y) {}
    ///
    ///   void operator () (const size_type i, value_type& update, const bool
    ///   final_pass) const {
    ///     update += m_x(i);
    ///     if (final_pass) {
    ///       m_y(i) = update;
    ///     }
    ///   }
    ///   void init (value_type& update) const {
    ///     update = 0;
    ///   }
    ///   void join (value_type& update, const value_type& input)
    ///   const {
    ///     update += input;
    ///   }
    ///
    /// private:
    ///   flare::Tensor<value_type*, execution_space> m_x;
    ///   flare::Tensor<value_type*, execution_space> m_y;
    /// };
    /// \endcode
    ///
    /// Here is an example of a functor which computes an <i>exclusive</i>
    /// scan of an array of \c int, in place.  In operator(), note both
    /// that the final_pass test and the update have switched places, and
    /// the use of a temporary.  If given an array [1, 2, 3, 4], this scan
    /// will overwrite that array with [0, 1, 3, 6].
    ///
    /// \code
    /// template<class SpaceType>
    /// class ExclScanFunctor {
    /// public:
    ///   using execution_space = SpaceType;
    ///   using value_type = int;
    ///   using size_type = typename SpaceType::size_type;
    ///
    ///   ExclScanFunctor (flare::Tensor<value_type*, execution_space> x) : x_ (x) {}
    ///
    ///   void operator () (const size_type i, value_type& update, const bool
    ///   final_pass) const {
    ///     const value_type x_i = x_(i);
    ///     if (final_pass) {
    ///       x_(i) = update;
    ///     }
    ///     update += x_i;
    ///   }
    ///   void init (value_type& update) const {
    ///     update = 0;
    ///   }
    ///   void join (value_type& update, const value_type& input)
    ///   const {
    ///     update += input;
    ///   }
    ///
    /// private:
    ///   flare::Tensor<value_type*, execution_space> x_;
    /// };
    /// \endcode
    ///
    /// Here is an example of a functor which builds on the above
    /// exclusive scan example, to compute an offsets array from a
    /// population count array, in place.  We assume that the pop count
    /// array has an extra entry at the end to store the final count.  If
    /// given an array [1, 2, 3, 4, 0], this scan will overwrite that
    /// array with [0, 1, 3, 6, 10].
    ///
    /// \code
    /// template<class SpaceType>
    /// class OffsetScanFunctor {
    /// public:
    ///   using execution_space = SpaceType;
    ///   using value_type = int;
    ///   using size_type = typename SpaceType::size_type;
    ///
    ///   // lastIndex_ is the last valid index (zero-based) of x.
    ///   // If x has length zero, then lastIndex_ won't be used anyway.
    ///   OffsetScanFunctor( flare::Tensor<value_type*, execution_space> x
    ///                    , flare::Tensor<value_type*, execution_space> y )
    ///      : m_x(x), m_y(y), last_index_ (x.dimension_0 () == 0 ? 0 :
    ///      x.dimension_0 () - 1)
    ///   {}
    ///
    ///   void operator () (const size_type i, int& update, const bool final_pass)
    ///   const {
    ///     if (final_pass) {
    ///       m_y(i) = update;
    ///     }
    ///     update += m_x(i);
    ///     // The last entry of m_y gets the final sum.
    ///     if (final_pass && i == last_index_) {
    ///       m_y(i+1) = update;
    // i/     }
    ///   }
    ///   void init (value_type& update) const {
    ///     update = 0;
    ///   }
    ///   void join (value_type& update, const value_type& input)
    ///   const {
    ///     update += input;
    ///   }
    ///
    /// private:
    ///   flare::Tensor<value_type*, execution_space> m_x;
    ///   flare::Tensor<value_type*, execution_space> m_y;
    ///   const size_type last_index_;
    /// };
    /// \endcode
    ///
    template<class ExecutionPolicy, class FunctorType,
            class Enable =
            std::enable_if_t<is_execution_policy<ExecutionPolicy>::value>>
    inline void parallel_scan(const std::string &str, const ExecutionPolicy &policy,
                              const FunctorType &functor) {
        uint64_t kpID = 0;
        ExecutionPolicy inner_policy = policy;
        flare::Tools::detail::begin_parallel_scan(inner_policy, functor, str, kpID);

        flare::detail::shared_allocation_tracking_disable();
        detail::ParallelScan<FunctorType, ExecutionPolicy> closure(functor,
                                                                   inner_policy);
        flare::detail::shared_allocation_tracking_enable();

        closure.execute();

        flare::Tools::detail::end_parallel_scan(inner_policy, functor, str, kpID);
    }

    template<class ExecutionPolicy, class FunctorType>
    inline void parallel_scan(
            const ExecutionPolicy &policy, const FunctorType &functor,
            std::enable_if_t<is_execution_policy<ExecutionPolicy>::value> * = nullptr) {
        ::flare::parallel_scan("", policy, functor);
    }

    template<class FunctorType>
    inline void parallel_scan(const std::string &str, const size_t work_count,
                              const FunctorType &functor) {
        using execution_space =
                typename flare::detail::FunctorPolicyExecutionSpace<FunctorType,
                        void>::execution_space;

        using policy = flare::RangePolicy<execution_space>;

        policy execution_policy(0, work_count);
        parallel_scan(str, execution_policy, functor);
    }

    template<class FunctorType>
    inline void parallel_scan(const size_t work_count, const FunctorType &functor) {
        ::flare::parallel_scan("", work_count, functor);
    }

    template<class ExecutionPolicy, class FunctorType, class ReturnType,
            class Enable =
            std::enable_if_t<is_execution_policy<ExecutionPolicy>::value>>
    inline void parallel_scan(const std::string &str, const ExecutionPolicy &policy,
                              const FunctorType &functor,
                              ReturnType &return_value) {
        uint64_t kpID = 0;
        ExecutionPolicy inner_policy = policy;
        flare::Tools::detail::begin_parallel_scan(inner_policy, functor, str, kpID);

        if constexpr (flare::is_tensor<ReturnType>::value) {
            flare::detail::shared_allocation_tracking_disable();
            detail::ParallelScanWithTotal<FunctorType, ExecutionPolicy,
                    typename ReturnType::value_type>
                    closure(functor, inner_policy, return_value);
            flare::detail::shared_allocation_tracking_enable();
            closure.execute();
        } else {
            flare::detail::shared_allocation_tracking_disable();
            flare::Tensor<ReturnType, flare::HostSpace> tensor(&return_value);
            detail::ParallelScanWithTotal<FunctorType, ExecutionPolicy, ReturnType>
                    closure(functor, inner_policy, tensor);
            flare::detail::shared_allocation_tracking_enable();
            closure.execute();
        }

        flare::Tools::detail::end_parallel_scan(inner_policy, functor, str, kpID);

        if (!flare::is_tensor<ReturnType>::value)
            policy.space().fence(
                    "flare::parallel_scan: fence due to result being a value, not a tensor");
    }

    template<class ExecutionPolicy, class FunctorType, class ReturnType>
    inline void parallel_scan(
            const ExecutionPolicy &policy, const FunctorType &functor,
            ReturnType &return_value,
            std::enable_if_t<is_execution_policy<ExecutionPolicy>::value> * = nullptr) {
        ::flare::parallel_scan("", policy, functor, return_value);
    }

    template<class FunctorType, class ReturnType>
    inline void parallel_scan(const std::string &str, const size_t work_count,
                              const FunctorType &functor,
                              ReturnType &return_value) {
        using execution_space =
                typename flare::detail::FunctorPolicyExecutionSpace<FunctorType,
                        void>::execution_space;

        using policy = flare::RangePolicy<execution_space>;

        policy execution_policy(0, work_count);
        parallel_scan(str, execution_policy, functor, return_value);
    }

    template<class FunctorType, class ReturnType>
    inline void parallel_scan(const size_t work_count, const FunctorType &functor,
                              ReturnType &return_value) {
        ::flare::parallel_scan("", work_count, functor, return_value);
    }

}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare::detail {

    template<class FunctorType,
            bool HasTeamShmemSize =
            has_member_team_shmem_size<FunctorType>::value,
            bool HasShmemSize = has_member_shmem_size<FunctorType>::value>
    struct FunctorTeamShmemSize {
        FLARE_INLINE_FUNCTION static size_t value(const FunctorType &, int) {
            return 0;
        }
    };

    template<class FunctorType>
    struct FunctorTeamShmemSize<FunctorType, true, false> {
        static inline size_t value(const FunctorType &f, int team_size) {
            return f.team_shmem_size(team_size);
        }
    };

    template<class FunctorType>
    struct FunctorTeamShmemSize<FunctorType, false, true> {
        static inline size_t value(const FunctorType &f, int team_size) {
            return f.shmem_size(team_size);
        }
    };

    template<class FunctorType>
    struct FunctorTeamShmemSize<FunctorType, true, true> {
        static inline size_t value(const FunctorType & /*f*/, int /*team_size*/) {
            flare::abort(
                    "Functor with both team_shmem_size and shmem_size defined is "
                    "not allowed");
            return 0;
        }
    };

}  // namespace flare::detail

#endif  // FLARE_CORE_PARALLEL_PARALLEL_H_
