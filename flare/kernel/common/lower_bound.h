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


#ifndef FLARE_KERNEL_COMMON_LOWER_BOUND_H_
#define FLARE_KERNEL_COMMON_LOWER_BOUND_H_

#include <flare/core/numeric_traits.h>
#include <flare/kernel/common/predicates.h>
#include <flare/kernel/common/simple_utility.h>

namespace flare::detail {

    /*! \brief Single-thread sequential lower-bound search

        \tparam ViewLike A flare::View or flare::detail::Iota
        \tparam Pred a binary predicate function
        \param view the view to search
        \param value the value to search for
        \param pred a binary predicate function
        \returns index of first element in view where pred(element, value) is false,
        or view.size if no such element exists

        At most view.size() predicate function calls
    */
    template<typename ViewLike,
            typename Pred = LT<typename ViewLike::non_const_value_type>>
    FLARE_INLINE_FUNCTION typename ViewLike::size_type
    lower_bound_sequential_thread(
            const ViewLike &view, const typename ViewLike::non_const_value_type &value,
            Pred pred = Pred()) {
        using size_type = typename ViewLike::size_type;
        static_assert(1 == ViewLike::rank,
                      "lower_bound_sequential_thread requires rank-1 views");
        static_assert(is_iota_v<ViewLike> || flare::is_view<ViewLike>::value,
                      "lower_bound_sequential_thread requires a "
                      "flare::detail::Iota or a flare::View");

        size_type i = 0;
        while (i < view.size() && pred(view(i), value)) {
            ++i;
        }
        return i;
    }

    /*! \brief Single-thread binary lower-bound search

        \tparam ViewLike A flare::View or flare::detail::Iota
        \tparam Pred a binary predicate function
        \param view the view to search
        \param value the value to search for
        \param pred a binary predicate function
        \returns index of first element in view where pred(element, value) is false,
        or view.size if no such element exists

        At most log2(view.size()) + 1 predicate function calls
    */
    template<typename ViewLike,
            typename Pred = LT<typename ViewLike::non_const_value_type>>
    FLARE_INLINE_FUNCTION typename ViewLike::size_type lower_bound_binary_thread(
            const ViewLike &view, const typename ViewLike::non_const_value_type &value,
            Pred pred = Pred()) {
        using size_type = typename ViewLike::size_type;
        static_assert(1 == ViewLike::rank,
                      "lower_bound_binary_thread requires rank-1 views");
        static_assert(is_iota_v<ViewLike> || flare::is_view<ViewLike>::value,
                      "lower_bound_binary_thread requires a "
                      "flare::detail::Iota or a flare::View");

        size_type lo = 0;
        size_type hi = view.size();
        while (lo < hi) {
            size_type mid = (lo + hi) / 2;
            const auto &ve = view(mid);
            if (pred(ve, value)) {  // mid satisfies predicate, look in higher half not
                // including mid
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

}  // namespace flare::detail

namespace flare {
    /*! \brief single-thread lower-bound search

        \tparam ViewLike A flare::View or flare::detail::Iota
        \tparam Pred a binary predicate function
        \param view the view to search
        \param value the value to search for
        \param pred a binary predicate function
        \returns index of first element in view where pred(element, value) is false,
        or view.size if no such element exists

        This minimizes the calls to predicate:
        for view.size() >= 8, this does a binary search, otherwise, a linear search
    */
    template<typename ViewLike,
            typename Pred = LT<typename ViewLike::non_const_value_type>>
    FLARE_INLINE_FUNCTION typename ViewLike::size_type

    lower_bound_thread(
            const ViewLike &view, const typename ViewLike::non_const_value_type &value,
            Pred pred = Pred()) {
        static_assert(1 == ViewLike::rank,
                      "lower_bound_thread requires rank-1 views");
        static_assert(flare::detail::is_iota_v<ViewLike> ||
                      flare::is_view<ViewLike>::value,
                      "lower_bound_thread requires a "
                      "flare::detail::Iota or a flare::View");
        /*
           sequential search makes on average 0.5 * view.size memory accesses
           binary search makes log2(view.size)+1 accesses

           log2(x) <= 0.5x roughly when x >= 8
        */
        if (view.size() >= 8) {
            return detail::lower_bound_binary_thread(view, value, pred);
        } else {
            return detail::lower_bound_sequential_thread(view, value, pred);
        }
    }
}
namespace flare::detail {

/*! \brief Team-collaborative sequential lower-bound search

    \tparam TeamMember the team policy member type
    \tparam ViewLike A flare::View or flare::Iota
    \tparam Pred The type of the predicate function to call

    \param handle The flare team handle
    \param view The view-like to search
    \param value The value to compare in the predicate
    \param lo The first index to search
    \param hi One-past the last index to search
    \param pred Apply pred(view(i), value)

    \returns To all team members, the smallest i for which pred(view(i), value)
   is false for i in [lo, hi), or hi if no such value

    Uses a single thread to call \c lower_bound_thread, and broadcasts that
    to all team members.
*/
    template<typename TeamMember, typename ViewLike,
            typename Pred = LT<typename ViewLike::non_const_value_type>>
    FLARE_INLINE_FUNCTION typename ViewLike::size_type lower_bound_single_team(
            const TeamMember &handle, const ViewLike &view,
            const typename ViewLike::non_const_value_type &value, Pred pred = Pred()) {
        typename ViewLike::size_type idx;
        flare::single(
                flare::PerTeam(handle),
                [&](typename ViewLike::size_type &lidx) {
                    lidx = flare::lower_bound_thread(view, value, pred);
                },
                idx);
        return idx;
    }

    /*! \brief Team-collaborative sequential lower-bound search

        \tparam TeamMember the team policy member type
        \tparam ViewLike A flare::View or flare::Iota
        \tparam Pred The type of the predicate function to call

        \param handle The flare team handle
        \param view The view-like to search
        \param value The value to compare in the predicate
        \param lo The first index to search
        \param hi One-past the last index to search
        \param pred Apply pred(view(i), value)F

        \returns To all team members, the smallest i for which pred(view(i), value)
       is false for i in [lo, hi), or hi if no such value

        Apply pred(view(i), value) for i in [lo, hi)
    */
    template<typename TeamMember, typename ViewLike,
            typename Pred = LT<typename ViewLike::non_const_value_type>>
    FLARE_INLINE_FUNCTION typename ViewLike::size_type lower_bound_sequential_team(
            const TeamMember &handle, const ViewLike &view,
            const typename ViewLike::non_const_value_type &value,
            typename ViewLike::size_type lo, typename ViewLike::size_type hi,
            Pred pred = Pred()) {
        using size_type = typename ViewLike::size_type;
        static_assert(1 == ViewLike::rank,
                      "lower_bound_sequential_team requires rank-1 views");
        static_assert(is_iota_v<ViewLike> || flare::is_view<ViewLike>::value,
                      "lower_bound_sequential_team requires a "
                      "flare::detail::Iota or a flare::View");

        if (lo == hi) {
            return hi;
        }
        size_type teamI;
        flare::parallel_reduce(
                flare::TeamThreadRange(handle, lo, hi),
                [&](const size_type &i, size_type &li) {
                    li = FLARE_MACRO_MIN(li, hi);
                    if (i < li) {  // no need to search higher than the smallest so far
                        if (!pred(view(i), value)) {  // look for the smallest index that does
                            // not satisfy
                            li = i;
                        }
                    }
                },
                flare::Min<size_type>(teamI));
        return teamI;
    }

    /*! \brief Team-collaborative sequential lower-bound search

        \tparam TeamMember the team policy member type
        \tparam ViewLike A flare::View or flare::Iota
        \tparam Pred The type of the predicate function to call

        \param handle The flare team handle
        \param view The view-like to search
        \param value The value to compare in the predicate
        \param pred Apply pred(view(i), value)

        \returns To all team members, the smallest i for which pred(view(i), value)
       is false or view.size() if no such value
    */
    template<typename TeamMember, typename ViewLike,
            typename Pred = LT<typename ViewLike::non_const_value_type>>
    FLARE_INLINE_FUNCTION typename ViewLike::size_type lower_bound_sequential_team(
            const TeamMember &handle, const ViewLike &view,
            const typename ViewLike::non_const_value_type &value, Pred pred = Pred()) {
        return lower_bound_sequential_team(handle, view, value, 0, view.size(), pred);
    }

/*! \brief A range for the k-ary lower bound search

    The RangeReducer will maximize the lower bound and
    minimize the upper bound
*/
    template<typename T>
    struct Range {
        T lb;  /// lower-bound
        T ub;  /// upper-bound

        FLARE_INLINE_FUNCTION
        Range() { init(); }

        FLARE_INLINE_FUNCTION
        constexpr Range(const T &_lb, const T &_ub) : lb(_lb), ub(_ub) {}

        FLARE_INLINE_FUNCTION
        void init() {
            lb = flare::experimental::finite_min_v<T>;  // will be max'd
            ub = flare::experimental::finite_max_v<T>;  // will be min'd
        }
    };

    /// \brief maximizes the lower bound, and minimizes the upper bound of a Range
    template<typename T, typename Space>
    struct RangeReducer {
        using reducer = RangeReducer;
        using value_type = Range<T>;
        using result_view_type =
                flare::View<Range<T> *, Space, flare::MemoryUnmanaged>;

    private:
        value_type &value;

    public:
        FLARE_INLINE_FUNCTION
        RangeReducer(value_type &value_) : value(value_) {}

        FLARE_INLINE_FUNCTION
        void join(value_type &dst, const value_type &src) const {
            dst.lb = FLARE_MACRO_MAX(dst.lb, src.lb);
            dst.ub = FLARE_MACRO_MIN(dst.ub, src.ub);
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const { val.init(); }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return value; }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return result_view_type(&value, 1); }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return true; }
    };

/*! \brief team-collaborative K-ary lower-bound search

    \tparam TeamMember the team policy member type
    \tparam ViewLike A flare::View or flare::Iota
    \tparam Pred the binary predicate function type

    Actually, K+1-ary, where K is the size of the team
    Split the view into k+1 segments at K points
    Evalute the predicate in parallel at each point and use a joint min-max
   parallel reduction:
      * The lower bound is after the max index where the predicate was true
      * The upper bound is no greater than the min index where the predicate was
   false Once there are fewer values left than threads in the team, switch to
   team sequential search
*/
    template<typename TeamMember, typename ViewLike,
            typename Pred = LT<typename ViewLike::non_const_value_type>>
    FLARE_INLINE_FUNCTION typename ViewLike::size_type lower_bound_kary_team(
            const TeamMember &handle, const ViewLike &view,
            const typename ViewLike::non_const_value_type &value, Pred pred = Pred()) {
        static_assert(1 == ViewLike::rank,
                      "lower_bound_kary_team requires rank-1 views");
        static_assert(is_iota_v<ViewLike> || flare::is_view<ViewLike>::value,
                      "lower_bound_kary_team requires a "
                      "flare::detail::Iota or a flare::View");

        using size_type = typename ViewLike::size_type;

        size_type lo = 0;
        size_type hi = view.size();
        while (lo < hi) {
            // if fewer than team_size elements left, just hit them all sequentially
            if (lo + handle.team_size() >= hi) {
                return lower_bound_sequential_team(handle, view, value, lo, hi, pred);
            }

            // otherwise, split the region up among threads
            size_type mid =
                    lo + (hi - lo) * (handle.team_rank() + 1) / (handle.team_size() + 1);
            auto ve = view(mid);

            // reduce across threads to figure out where the new search bounds are
            // if a thread satisfies the predicate, the first element that does not
            // satisfy must be after that thread's search point. we want the max such
            // point across all threads if a thread does not satisfy the predicate, the
            // first element that does not satisfy must be before or equal. we want the
            // min such point across all threads
            Range<size_type> teamRange;
            flare::parallel_reduce(
                    flare::TeamThreadRange(handle, 0, handle.team_size()),
                    [&](const int &, Range<size_type> &lr) {
                        lr.lb = FLARE_MACRO_MAX(lo, lr.lb);  // no lower than lo
                        lr.ub = FLARE_MACRO_MIN(hi, lr.ub);  // no higher than hi
                        // if pred(view(mid), value), then the lower bound is above this
                        if (pred(ve, value)) {
                            lr.lb = mid + 1;
                        } else {  // otherwise the lower bound is no larger than this
                            lr.ub = mid;
                        }
                    },
                    RangeReducer<size_type, typename ViewLike::device_type>(teamRange));

            // next iteration, search in the newly-discovered window
            hi = teamRange.ub;
            lo = teamRange.lb;
        }
        return lo;
    }

}  // namespace flare::detail
namespace flare {
    /*! \brief Team-collaborative lower-bound search

        \tparam TeamMember the team policy member type the flare team handle
        \tparam View the type of view
        \tparam Pred the type of the predicate

        \param handle a flare team handle
        \param view a flare::View to search
        \param value the value to search for
        \param pred the predicate to test entries in the view

        \returns The smallest i in range [0, view.size()) for which pred(view(i),
       value) is not true, or view.size() if no such `i` exists

        default pred is `element < value`, i.e. return the index to the first
       element in the view that does not satisfy `element < value`. For well-ordered
       types this is the first element where element >= value

        Pred should be a binary function comparing two `typename
       View::non_const_value_type`
    */
    template<typename TeamMember, typename ViewLike,
            typename Pred = LT<typename ViewLike::non_const_value_type>>
    FLARE_INLINE_FUNCTION typename ViewLike::size_type lower_bound_team(
            const TeamMember &handle, const ViewLike &view,
            const typename ViewLike::non_const_value_type &value, Pred pred = Pred()) {
        static_assert(1 == ViewLike::rank, "lower_bound_team requires rank-1 views");
        static_assert(flare::detail::is_iota_v<ViewLike> ||
                      flare::is_view<ViewLike>::value,
                      "lower_bound_team requires a "
                      "flare::detail::Iota or a flare::View");

        /* kary search is A = (k-1) * (logk(view.size()) + 1) accesses

           sequential search is B = view.size() accesses

            A < B is true ruoughly when view.size() > 3 * k
        */
        if (view.size() > 3 * size_t(handle.team_size())) {
            return detail::lower_bound_kary_team(handle, view, value, pred);
        } else {
            return detail::lower_bound_sequential_team(handle, view, value, pred);
        }
    }

}  // namespace flare

#endif  // FLARE_KERNEL_COMMON_LOWER_BOUND_H_
