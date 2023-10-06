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

#ifndef FLARE_CORE_PARALLEL_PARALLEL_REDUCE_H_
#define FLARE_CORE_PARALLEL_PARALLEL_REDUCE_H_

#include <flare/core/reduction_identity.h>
#include <flare/core/tensor/view.h>
#include <flare/core/common/functor_analysis.h>
#include <flare/core/profile/tools_generic.h>
#include <type_traits>
#include <iostream>

namespace flare {

    template<class Scalar, class Space>
    struct Sum {
    public:
        // Required
        using reducer = Sum<Scalar, Space>;
        using value_type = std::remove_cv_t<Scalar>;
        static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        Sum(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        Sum(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const { dest += src; }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val = reduction_identity<value_type>::sum();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename... Properties>
    Sum(View<Scalar, Properties...> const &)
    ->Sum<Scalar, typename View<Scalar, Properties...>::memory_space>;

    template<class Scalar, class Space>
    struct Prod {
    public:
        // Required
        using reducer = Prod<Scalar, Space>;
        using value_type = std::remove_cv_t<Scalar>;
        static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        Prod(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        Prod(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const { dest *= src; }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val = reduction_identity<value_type>::prod();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename... Properties>
    Prod(View<Scalar, Properties...> const &)
    ->Prod<Scalar, typename View<Scalar, Properties...>::memory_space>;

    template<class Scalar, class Space>
    struct Min {
    public:
        // Required
        using reducer = Min<Scalar, Space>;
        using value_type = std::remove_cv_t<Scalar>;
        static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        Min(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        Min(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (src < dest) dest = src;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val = reduction_identity<value_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename... Properties>
    Min(View<Scalar, Properties...> const &)
    ->Min<Scalar, typename View<Scalar, Properties...>::memory_space>;

    template<class Scalar, class Space>
    struct Max {
    public:
        // Required
        using reducer = Max<Scalar, Space>;
        using value_type = std::remove_cv_t<Scalar>;
        static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        Max(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        Max(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (src > dest) dest = src;
        }

        // Required
        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val = reduction_identity<value_type>::max();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename... Properties>
    Max(View<Scalar, Properties...> const &)
    ->Max<Scalar, typename View<Scalar, Properties...>::memory_space>;

    template<class Scalar, class Space>
    struct LAnd {
    public:
        // Required
        using reducer = LAnd<Scalar, Space>;
        using value_type = std::remove_cv_t<Scalar>;
        static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        LAnd(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        LAnd(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            dest = dest && src;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val = reduction_identity<value_type>::land();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename... Properties>
    LAnd(View<Scalar, Properties...> const &)
    ->LAnd<Scalar, typename View<Scalar, Properties...>::memory_space>;

    template<class Scalar, class Space>
    struct LOr {
    public:
        // Required
        using reducer = LOr<Scalar, Space>;
        using value_type = std::remove_cv_t<Scalar>;
        static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        LOr(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        LOr(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            dest = dest || src;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val = reduction_identity<value_type>::lor();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename... Properties>
    LOr(View<Scalar, Properties...> const &)
    ->LOr<Scalar, typename View<Scalar, Properties...>::memory_space>;

    template<class Scalar, class Space>
    struct BAnd {
    public:
        // Required
        using reducer = BAnd<Scalar, Space>;
        using value_type = std::remove_cv_t<Scalar>;
        static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        BAnd(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        BAnd(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            dest = dest & src;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val = reduction_identity<value_type>::band();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename... Properties>
    BAnd(View<Scalar, Properties...> const &)
    ->BAnd<Scalar, typename View<Scalar, Properties...>::memory_space>;

    template<class Scalar, class Space>
    struct BOr {
    public:
        // Required
        using reducer = BOr<Scalar, Space>;
        using value_type = std::remove_cv_t<Scalar>;
        static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        BOr(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        BOr(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            dest = dest | src;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val = reduction_identity<value_type>::bor();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename... Properties>
    BOr(View<Scalar, Properties...> const &)
    ->BOr<Scalar, typename View<Scalar, Properties...>::memory_space>;

    template<class Scalar, class Index>
    struct ValLocScalar {
        Scalar val;
        Index loc;
    };

    template<class Scalar, class Index, class Space>
    struct MinLoc {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        using index_type = std::remove_cv_t<Index>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);

    public:
        // Required
        using reducer = MinLoc<Scalar, Index, Space>;
        using value_type = ValLocScalar<scalar_type, index_type>;

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        MinLoc(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        MinLoc(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (src.val < dest.val) dest = src;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.val = reduction_identity<scalar_type>::min();
            val.loc = reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename Index, typename... Properties>
    MinLoc(View<ValLocScalar<Scalar, Index>, Properties...> const &)
    ->MinLoc<Scalar, Index,
            typename View<ValLocScalar<Scalar, Index>,
                    Properties...>::memory_space>;

    template<class Scalar, class Index, class Space>
    struct MaxLoc {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        using index_type = std::remove_cv_t<Index>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);

    public:
        // Required
        using reducer = MaxLoc<Scalar, Index, Space>;
        using value_type = ValLocScalar<scalar_type, index_type>;

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        MaxLoc(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        MaxLoc(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (src.val > dest.val) dest = src;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.val = reduction_identity<scalar_type>::max();
            val.loc = reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename Index, typename... Properties>
    MaxLoc(View<ValLocScalar<Scalar, Index>, Properties...> const &)
    ->MaxLoc<Scalar, Index,
            typename View<ValLocScalar<Scalar, Index>,
                    Properties...>::memory_space>;

    template<class Scalar>
    struct MinMaxScalar {
        Scalar min_val, max_val;
    };

    template<class Scalar, class Space>
    struct MinMax {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);

    public:
        // Required
        using reducer = MinMax<Scalar, Space>;
        using value_type = MinMaxScalar<scalar_type>;

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        MinMax(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        MinMax(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (src.min_val < dest.min_val) {
                dest.min_val = src.min_val;
            }
            if (src.max_val > dest.max_val) {
                dest.max_val = src.max_val;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.max_val = reduction_identity<scalar_type>::max();
            val.min_val = reduction_identity<scalar_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename... Properties>
    MinMax(View<MinMaxScalar<Scalar>, Properties...> const &)
    ->MinMax<Scalar,
            typename View<MinMaxScalar<Scalar>, Properties...>::memory_space>;

    template<class Scalar, class Index>
    struct MinMaxLocScalar {
        Scalar min_val, max_val;
        Index min_loc, max_loc;
    };

    template<class Scalar, class Index, class Space>
    struct MinMaxLoc {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        using index_type = std::remove_cv_t<Index>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);

    public:
        // Required
        using reducer = MinMaxLoc<Scalar, Index, Space>;
        using value_type = MinMaxLocScalar<scalar_type, index_type>;

        using result_view_type = flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        MinMaxLoc(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        MinMaxLoc(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (src.min_val < dest.min_val) {
                dest.min_val = src.min_val;
                dest.min_loc = src.min_loc;
            }
            if (src.max_val > dest.max_val) {
                dest.max_val = src.max_val;
                dest.max_loc = src.max_loc;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.max_val = reduction_identity<scalar_type>::max();
            val.min_val = reduction_identity<scalar_type>::min();
            val.max_loc = reduction_identity<index_type>::min();
            val.min_loc = reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename Index, typename... Properties>
    MinMaxLoc(View<MinMaxLocScalar<Scalar, Index>, Properties...> const &)
    ->MinMaxLoc<Scalar, Index,
            typename View<MinMaxLocScalar<Scalar, Index>,
                    Properties...>::memory_space>;

// --------------------------------------------------
// reducers added to support std algorithms
// --------------------------------------------------

//
// MaxFirstLoc
//
    template<class Scalar, class Index, class Space>
    struct MaxFirstLoc {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        using index_type = std::remove_cv_t<Index>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer = MaxFirstLoc<Scalar, Index, Space>;
        using value_type = ::flare::ValLocScalar<scalar_type, index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        MaxFirstLoc(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        MaxFirstLoc(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (dest.val < src.val) {
                dest = src;
            } else if (!(src.val < dest.val)) {
                dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.val = reduction_identity<scalar_type>::max();
            val.loc = reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename Index, typename... Properties>
    MaxFirstLoc(View<ValLocScalar<Scalar, Index>, Properties...> const &)
    ->MaxFirstLoc<Scalar, Index,
            typename View<ValLocScalar<Scalar, Index>,
                    Properties...>::memory_space>;

//
// MaxFirstLocCustomComparator
// recall that comp(a,b) returns true is a < b
//
    template<class Scalar, class Index, class ComparatorType, class Space>
    struct MaxFirstLocCustomComparator {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        using index_type = std::remove_cv_t<Index>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer =
                MaxFirstLocCustomComparator<Scalar, Index, ComparatorType, Space>;
        using value_type = ::flare::ValLocScalar<scalar_type, index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;
        ComparatorType m_comp;

    public:
        FLARE_INLINE_FUNCTION
        MaxFirstLocCustomComparator(value_type &value_, ComparatorType comp_)
                : value(&value_), references_scalar_v(true), m_comp(comp_) {}

        FLARE_INLINE_FUNCTION
        MaxFirstLocCustomComparator(const result_view_type &value_,
                                    ComparatorType comp_)
                : value(value_), references_scalar_v(false), m_comp(comp_) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (m_comp(dest.val, src.val)) {
                dest = src;
            } else if (!m_comp(src.val, dest.val)) {
                dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.val = reduction_identity<scalar_type>::max();
            val.loc = reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename Index, typename ComparatorType,
            typename... Properties>
    MaxFirstLocCustomComparator(
            View<ValLocScalar<Scalar, Index>, Properties...> const &, ComparatorType)
    ->MaxFirstLocCustomComparator<Scalar, Index, ComparatorType,
            typename View<ValLocScalar<Scalar, Index>,
                    Properties...>::memory_space>;

//
// MinFirstLoc
//
    template<class Scalar, class Index, class Space>
    struct MinFirstLoc {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        using index_type = std::remove_cv_t<Index>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer = MinFirstLoc<Scalar, Index, Space>;
        using value_type = ::flare::ValLocScalar<scalar_type, index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        MinFirstLoc(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        MinFirstLoc(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (src.val < dest.val) {
                dest = src;
            } else if (!(dest.val < src.val)) {
                dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.val = reduction_identity<scalar_type>::min();
            val.loc = reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename Index, typename... Properties>
    MinFirstLoc(View<ValLocScalar<Scalar, Index>, Properties...> const &)
    ->MinFirstLoc<Scalar, Index,
            typename View<ValLocScalar<Scalar, Index>,
                    Properties...>::memory_space>;

//
// MinFirstLocCustomComparator
// recall that comp(a,b) returns true is a < b
//
    template<class Scalar, class Index, class ComparatorType, class Space>
    struct MinFirstLocCustomComparator {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        using index_type = std::remove_cv_t<Index>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer =
                MinFirstLocCustomComparator<Scalar, Index, ComparatorType, Space>;
        using value_type = ::flare::ValLocScalar<scalar_type, index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;
        ComparatorType m_comp;

    public:
        FLARE_INLINE_FUNCTION
        MinFirstLocCustomComparator(value_type &value_, ComparatorType comp_)
                : value(&value_), references_scalar_v(true), m_comp(comp_) {}

        FLARE_INLINE_FUNCTION
        MinFirstLocCustomComparator(const result_view_type &value_,
                                    ComparatorType comp_)
                : value(value_), references_scalar_v(false), m_comp(comp_) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (m_comp(src.val, dest.val)) {
                dest = src;
            } else if (!m_comp(dest.val, src.val)) {
                dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.val = reduction_identity<scalar_type>::min();
            val.loc = reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename Index, typename ComparatorType,
            typename... Properties>
    MinFirstLocCustomComparator(
            View<ValLocScalar<Scalar, Index>, Properties...> const &, ComparatorType)
    ->MinFirstLocCustomComparator<Scalar, Index, ComparatorType,
            typename View<ValLocScalar<Scalar, Index>,
                    Properties...>::memory_space>;

//
// MinMaxFirstLastLoc
//
    template<class Scalar, class Index, class Space>
    struct MinMaxFirstLastLoc {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        using index_type = std::remove_cv_t<Index>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer = MinMaxFirstLastLoc<Scalar, Index, Space>;
        using value_type = ::flare::MinMaxLocScalar<scalar_type, index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        MinMaxFirstLastLoc(value_type &value_)
                : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        MinMaxFirstLastLoc(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (src.min_val < dest.min_val) {
                dest.min_val = src.min_val;
                dest.min_loc = src.min_loc;
            } else if (!(dest.min_val < src.min_val)) {
                dest.min_loc = (src.min_loc < dest.min_loc) ? src.min_loc : dest.min_loc;
            }

            if (dest.max_val < src.max_val) {
                dest.max_val = src.max_val;
                dest.max_loc = src.max_loc;
            } else if (!(src.max_val < dest.max_val)) {
                dest.max_loc = (src.max_loc > dest.max_loc) ? src.max_loc : dest.max_loc;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.max_val = ::flare::reduction_identity<scalar_type>::max();
            val.min_val = ::flare::reduction_identity<scalar_type>::min();
            val.max_loc = ::flare::reduction_identity<index_type>::max();
            val.min_loc = ::flare::reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename Index, typename... Properties>
    MinMaxFirstLastLoc(View<MinMaxLocScalar<Scalar, Index>, Properties...> const &)
    ->MinMaxFirstLastLoc<Scalar, Index,
            typename View<MinMaxLocScalar<Scalar, Index>,
                    Properties...>::memory_space>;

//
// MinMaxFirstLastLocCustomComparator
// recall that comp(a,b) returns true is a < b
//
    template<class Scalar, class Index, class ComparatorType, class Space>
    struct MinMaxFirstLastLocCustomComparator {
    private:
        using scalar_type = std::remove_cv_t<Scalar>;
        using index_type = std::remove_cv_t<Index>;
        static_assert(!std::is_pointer_v<scalar_type> &&
                      !std::is_array_v<scalar_type>);
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer =
                MinMaxFirstLastLocCustomComparator<Scalar, Index, ComparatorType, Space>;
        using value_type = ::flare::MinMaxLocScalar<scalar_type, index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;
        ComparatorType m_comp;

    public:
        FLARE_INLINE_FUNCTION
        MinMaxFirstLastLocCustomComparator(value_type &value_, ComparatorType comp_)
                : value(&value_), references_scalar_v(true), m_comp(comp_) {}

        FLARE_INLINE_FUNCTION
        MinMaxFirstLastLocCustomComparator(const result_view_type &value_,
                                           ComparatorType comp_)
                : value(value_), references_scalar_v(false), m_comp(comp_) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            if (m_comp(src.min_val, dest.min_val)) {
                dest.min_val = src.min_val;
                dest.min_loc = src.min_loc;
            } else if (!m_comp(dest.min_val, src.min_val)) {
                dest.min_loc = (src.min_loc < dest.min_loc) ? src.min_loc : dest.min_loc;
            }

            if (m_comp(dest.max_val, src.max_val)) {
                dest.max_val = src.max_val;
                dest.max_loc = src.max_loc;
            } else if (!m_comp(src.max_val, dest.max_val)) {
                dest.max_loc = (src.max_loc > dest.max_loc) ? src.max_loc : dest.max_loc;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.max_val = ::flare::reduction_identity<scalar_type>::max();
            val.min_val = ::flare::reduction_identity<scalar_type>::min();
            val.max_loc = ::flare::reduction_identity<index_type>::max();
            val.min_loc = ::flare::reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Scalar, typename Index, typename ComparatorType,
            typename... Properties>
    MinMaxFirstLastLocCustomComparator(
            View<MinMaxLocScalar<Scalar, Index>, Properties...> const &, ComparatorType)
    ->MinMaxFirstLastLocCustomComparator<
            Scalar, Index, ComparatorType,
            typename View<MinMaxLocScalar<Scalar, Index>,
                    Properties...>::memory_space>;

//
// FirstLoc
//
    template<class Index>
    struct FirstLocScalar {
        Index min_loc_true;
    };

    template<class Index, class Space>
    struct FirstLoc {
    private:
        using index_type = std::remove_cv_t<Index>;
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer = FirstLoc<Index, Space>;
        using value_type = FirstLocScalar<index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        FirstLoc(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        FirstLoc(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            dest.min_loc_true = (src.min_loc_true < dest.min_loc_true)
                                ? src.min_loc_true
                                : dest.min_loc_true;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.min_loc_true = ::flare::reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Index, typename... Properties>
    FirstLoc(View<FirstLocScalar<Index>, Properties...> const &)
    ->FirstLoc<Index, typename View<FirstLocScalar<Index>,
            Properties...>::memory_space>;

//
// LastLoc
//
    template<class Index>
    struct LastLocScalar {
        Index max_loc_true;
    };

    template<class Index, class Space>
    struct LastLoc {
    private:
        using index_type = std::remove_cv_t<Index>;
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer = LastLoc<Index, Space>;
        using value_type = LastLocScalar<index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        LastLoc(value_type &value_) : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        LastLoc(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            dest.max_loc_true = (src.max_loc_true > dest.max_loc_true)
                                ? src.max_loc_true
                                : dest.max_loc_true;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.max_loc_true = ::flare::reduction_identity<index_type>::max();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Index, typename... Properties>
    LastLoc(View<LastLocScalar<Index>, Properties...> const &)
    ->LastLoc<Index,
            typename View<LastLocScalar<Index>, Properties...>::memory_space>;

    template<class Index>
    struct StdIsPartScalar {
        Index max_loc_true, min_loc_false;
    };

//
// StdIsPartitioned
//
    template<class Index, class Space>
    struct StdIsPartitioned {
    private:
        using index_type = std::remove_cv_t<Index>;
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer = StdIsPartitioned<Index, Space>;
        using value_type = StdIsPartScalar<index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        StdIsPartitioned(value_type &value_)
                : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        StdIsPartitioned(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            dest.max_loc_true = (dest.max_loc_true < src.max_loc_true)
                                ? src.max_loc_true
                                : dest.max_loc_true;

            dest.min_loc_false = (dest.min_loc_false < src.min_loc_false)
                                 ? dest.min_loc_false
                                 : src.min_loc_false;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.max_loc_true = ::flare::reduction_identity<index_type>::max();
            val.min_loc_false = ::flare::reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Index, typename... Properties>
    StdIsPartitioned(View<StdIsPartScalar<Index>, Properties...> const &)
    ->StdIsPartitioned<Index, typename View<StdIsPartScalar<Index>,
            Properties...>::memory_space>;

    template<class Index>
    struct StdPartPointScalar {
        Index min_loc_false;
    };

//
// StdPartitionPoint
//
    template<class Index, class Space>
    struct StdPartitionPoint {
    private:
        using index_type = std::remove_cv_t<Index>;
        static_assert(std::is_integral_v<index_type>);

    public:
        // Required
        using reducer = StdPartitionPoint<Index, Space>;
        using value_type = StdPartPointScalar<index_type>;

        using result_view_type = ::flare::View<value_type, Space>;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
        FLARE_INLINE_FUNCTION
        StdPartitionPoint(value_type &value_)
                : value(&value_), references_scalar_v(true) {}

        FLARE_INLINE_FUNCTION
        StdPartitionPoint(const result_view_type &value_)
                : value(value_), references_scalar_v(false) {}

        // Required
        FLARE_INLINE_FUNCTION
        void join(value_type &dest, const value_type &src) const {
            dest.min_loc_false = (dest.min_loc_false < src.min_loc_false)
                                 ? dest.min_loc_false
                                 : src.min_loc_false;
        }

        FLARE_INLINE_FUNCTION
        void init(value_type &val) const {
            val.min_loc_false = ::flare::reduction_identity<index_type>::min();
        }

        FLARE_INLINE_FUNCTION
        value_type &reference() const { return *value.data(); }

        FLARE_INLINE_FUNCTION
        result_view_type view() const { return value; }

        FLARE_INLINE_FUNCTION
        bool references_scalar() const { return references_scalar_v; }
    };

    template<typename Index, typename... Properties>
    StdPartitionPoint(View<StdPartPointScalar<Index>, Properties...> const &)
    ->StdPartitionPoint<Index, typename View<StdPartPointScalar<Index>,
            Properties...>::memory_space>;

}  // namespace flare
namespace flare {
    namespace detail {

        template<typename FunctorType, typename FunctorAnalysisReducerType,
                typename Enable>
        class CombinedFunctorReducer {
        public:
            using functor_type = FunctorType;
            using reducer_type = FunctorAnalysisReducerType;

            CombinedFunctorReducer(const FunctorType &functor,
                                   const FunctorAnalysisReducerType &reducer)
                    : m_functor(functor), m_reducer(reducer) {}

            FLARE_FUNCTION const FunctorType &get_functor() const { return m_functor; }

            FLARE_FUNCTION const FunctorAnalysisReducerType &get_reducer() const {
                return m_reducer;
            }

        private:
            FunctorType m_functor;
            FunctorAnalysisReducerType m_reducer;
        };

        template<typename FunctorType, typename FunctorAnalysisReducerType>
        class CombinedFunctorReducer<
                FunctorType, FunctorAnalysisReducerType,
                std::enable_if_t<std::is_same_v<
                        FunctorType, typename FunctorAnalysisReducerType::functor_type>>> {
        public:
            using functor_type = FunctorType;
            using reducer_type = FunctorAnalysisReducerType;

            CombinedFunctorReducer(const FunctorType &functor,
                                   const FunctorAnalysisReducerType &)
                    : m_reducer(functor) {}

            FLARE_FUNCTION const FunctorType &get_functor() const {
                return m_reducer.get_functor();
            }

            FLARE_FUNCTION const FunctorAnalysisReducerType &get_reducer() const {
                return m_reducer;
            }

        private:
            FunctorAnalysisReducerType m_reducer;
        };

        template<class T, class ReturnType, class ValueTraits>
        struct ParallelReduceReturnValue;

        template<class ReturnType, class FunctorType>
        struct ParallelReduceReturnValue<
                std::enable_if_t<flare::is_view<ReturnType>::value>, ReturnType,
                FunctorType> {
            using return_type = ReturnType;
            using reducer_type = InvalidType;

            using value_type_scalar = typename return_type::value_type;
            using value_type_array = typename return_type::value_type *const;

            using value_type = std::conditional_t<return_type::rank == 0,
                    value_type_scalar, value_type_array>;

            static return_type &return_value(ReturnType &return_val, const FunctorType &) {
                return return_val;
            }
        };

        template<class ReturnType, class FunctorType>
        struct ParallelReduceReturnValue<
                std::enable_if_t<!flare::is_view<ReturnType>::value &&
                                 (!std::is_array<ReturnType>::value &&
                                  !std::is_pointer<ReturnType>::value) &&
                                 !flare::is_reducer<ReturnType>::value>,
                ReturnType, FunctorType> {
            using return_type =
                    flare::View<ReturnType, flare::HostSpace, flare::MemoryUnmanaged>;

            using reducer_type = InvalidType;

            using value_type = typename return_type::value_type;

            static return_type return_value(ReturnType &return_val, const FunctorType &) {
                return return_type(&return_val);
            }
        };

        template<class ReturnType, class FunctorType>
        struct ParallelReduceReturnValue<
                std::enable_if_t<(std::is_array<ReturnType>::value ||
                                  std::is_pointer<ReturnType>::value)>,
                ReturnType, FunctorType> {
            using return_type = flare::View<std::remove_const_t<ReturnType>,
                    flare::HostSpace, flare::MemoryUnmanaged>;

            using reducer_type = InvalidType;

            using value_type = typename return_type::value_type[];

            static return_type return_value(ReturnType &return_val,
                                            const FunctorType &functor) {
                if (std::is_array<ReturnType>::value)
                    return return_type(return_val);
                else
                    return return_type(return_val, functor.value_count);
            }
        };

        template<class ReturnType, class FunctorType>
        struct ParallelReduceReturnValue<
                std::enable_if_t<flare::is_reducer<ReturnType>::value>, ReturnType,
                FunctorType> {
            using return_type = typename ReturnType::result_view_type;
            using reducer_type = ReturnType;
            using value_type = typename return_type::value_type;

            static auto return_value(ReturnType &return_val, const FunctorType &) {
                return return_val.view();
            }
        };

        template<class T, class ReturnType, class FunctorType>
        struct ParallelReducePolicyType;

        template<class PolicyType, class FunctorType>
        struct ParallelReducePolicyType<
                std::enable_if_t<flare::is_execution_policy<PolicyType>::value>,
                PolicyType, FunctorType> {
            using policy_type = PolicyType;

            static PolicyType policy(const PolicyType &policy_) { return policy_; }
        };

        template<class PolicyType, class FunctorType>
        struct ParallelReducePolicyType<
                std::enable_if_t<std::is_integral<PolicyType>::value>, PolicyType,
                FunctorType> {
            using execution_space =
                    typename detail::FunctorPolicyExecutionSpace<FunctorType,
                            void>::execution_space;

            using policy_type = flare::RangePolicy<execution_space>;

            static policy_type policy(const PolicyType &policy_) {
                return policy_type(0, policy_);
            }
        };

        template<class FunctorType, class ExecPolicy, class ValueType,
                class ExecutionSpace>
        struct ParallelReduceFunctorType {
            using functor_type = FunctorType;

            static const functor_type &functor(const functor_type &functor) {
                return functor;
            }
        };

        template<class PolicyType, class FunctorType, class ReturnType>
        struct ParallelReduceAdaptor {
            using return_value_adapter =
                    detail::ParallelReduceReturnValue<void, ReturnType, FunctorType>;

            static inline void execute_impl(const std::string &label,
                                            const PolicyType &policy,
                                            const FunctorType &functor,
                                            ReturnType &return_value) {
                using PassedReducerType = typename return_value_adapter::reducer_type;
                uint64_t kpID = 0;

                PolicyType inner_policy = policy;
                flare::Tools::detail::begin_parallel_reduce<PassedReducerType>(
                        inner_policy, functor, label, kpID);

                using ReducerSelector =
                        flare::detail::if_c<std::is_same<InvalidType, PassedReducerType>::value,
                                FunctorType, PassedReducerType>;
                using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                        PolicyType, typename ReducerSelector::type,
                        typename return_value_adapter::value_type>;
                flare::detail::shared_allocation_tracking_disable();
                CombinedFunctorReducer functor_reducer(
                        functor, typename Analysis::Reducer(
                                ReducerSelector::select(functor, return_value)));

                // FIXME Remove "Wrapper" once all backends implement the new interface
                detail::ParallelReduce<decltype(functor_reducer), PolicyType,
                        typename detail::FunctorPolicyExecutionSpace<
                                FunctorType, PolicyType>::execution_space>
                        closure(functor_reducer, inner_policy,
                                return_value_adapter::return_value(return_value, functor));
                flare::detail::shared_allocation_tracking_enable();
                closure.execute();

                flare::Tools::detail::end_parallel_reduce<PassedReducerType>(
                        inner_policy, functor, label, kpID);
            }

            static constexpr bool is_array_reduction =
                    detail::FunctorAnalysis<
                            detail::FunctorPatternInterface::REDUCE, PolicyType, FunctorType,
                            typename return_value_adapter::value_type>::StaticValueSize == 0;

            template<typename Dummy = ReturnType>
            static inline std::enable_if_t<!(is_array_reduction &&
                                             std::is_pointer<Dummy>::value)>
            execute(const std::string &label, const PolicyType &policy,
                    const FunctorType &functor, ReturnType &return_value) {
                execute_impl(label, policy, functor, return_value);
            }
        };
    }  // namespace detail

//----------------------------------------------------------------------------

/*! \fn void parallel_reduce(label,policy,functor,return_argument)
    \brief Perform a parallel reduction.
    \param label An optional Label giving the call name. Must be able to
   construct a std::string from the argument. \param policy A flare Execution
   Policy, such as an integer, a RangePolicy or a TeamPolicy. \param functor A
   functor with a reduction operator, and optional init, join and final
   functions. \param return_argument A return argument which can be a scalar, a
   View, or a ReducerStruct. This argument can be left out if the functor has a
   final function.
*/

// Parallel Reduce Blocking behavior

    namespace detail {
        template<typename T>
        struct ReducerHasTestReferenceFunction {
            template<typename E>
            static std::true_type test_func(decltype(&E::references_scalar));

            template<typename E>
            static std::false_type test_func(...);

            enum {
                value = std::is_same<std::true_type, decltype(test_func<T>(nullptr))>::value
            };
        };

        template<class ExecutionSpace, class Arg>
        constexpr std::enable_if_t<
                // constraints only necessary because SFINAE lacks subsumption
                !ReducerHasTestReferenceFunction<Arg>::value &&
                !flare::is_view<Arg>::value,
                // return type:
                bool>
        parallel_reduce_needs_fence(ExecutionSpace const &, Arg const &) {
            return true;
        }

        template<class ExecutionSpace, class Reducer>
        constexpr std::enable_if_t<
                // equivalent to:
                // (requires (Reducer const& r) {
                //   { reducer.references_scalar() } -> std::convertible_to<bool>;
                // })
                ReducerHasTestReferenceFunction<Reducer>::value,
                // return type:
                bool>
        parallel_reduce_needs_fence(ExecutionSpace const &, Reducer const &reducer) {
            return reducer.references_scalar();
        }

        template<class ExecutionSpace, class ViewLike>
        constexpr std::enable_if_t<
                // requires flare::ViewLike<ViewLike>
                flare::is_view<ViewLike>::value,
                // return type:
                bool>
        parallel_reduce_needs_fence(ExecutionSpace const &, ViewLike const &) {
            return false;
        }

        template<class ExecutionSpace, class... Args>
        struct ParallelReduceFence {
            template<class... ArgsDeduced>
            static void fence(const ExecutionSpace &ex, const std::string &name,
                              ArgsDeduced &&... args) {
                if (detail::parallel_reduce_needs_fence(ex, (ArgsDeduced &&) args...)) {
                    ex.fence(name);
                }
            }
        };

    }  // namespace detail

/** \brief  Parallel reduction
 *
 * parallel_reduce performs parallel reductions with arbitrary functions - i.e.
 * it is not solely data based. The call expects up to 4 arguments:
 *
 *
 * Example of a parallel_reduce functor for a POD (plain old data) value type:
 * \code
 *  class FunctorType { // For POD value type
 *  public:
 *    using execution_space = ...;
 *    using value_type = <podType>;
 *    void operator()( <intType> iwork , <podType> & update ) const ;
 *    void init( <podType> & update ) const ;
 *    void join(       <podType> & update ,
 *               const <podType> & input ) const ;
 *
 *    void final( <podType> & update ) const ;
 *  };
 * \endcode
 *
 * Example of a parallel_reduce functor for an array of POD (plain old data)
 * values:
 * \code
 *  class FunctorType { // For array of POD value
 *  public:
 *    using execution_space = ...;
 *    using value_type = <podType>[];
 *    void operator()( <intType> , <podType> update[] ) const ;
 *    void init( <podType> update[] ) const ;
 *    void join(       <podType> update[] ,
 *               const <podType> input[] ) const ;
 *
 *    void final( <podType> update[] ) const ;
 *  };
 * \endcode
 */

// ReturnValue is scalar or array: take by reference

    template<class PolicyType, class FunctorType, class ReturnType>
    inline std::enable_if_t<flare::is_execution_policy<PolicyType>::value &&
                            !(flare::is_view<ReturnType>::value ||
                              flare::is_reducer<ReturnType>::value ||
                              std::is_pointer<ReturnType>::value)>
    parallel_reduce(const std::string &label, const PolicyType &policy,
                    const FunctorType &functor, ReturnType &return_value) {
        static_assert(
                !std::is_const<ReturnType>::value,
                "A const reduction result type is only allowed for a View, pointer or "
                "reducer return type!");

        detail::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
                label, policy, functor, return_value);
        detail::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
        fence(
                policy.space(),
                "flare::parallel_reduce: fence due to result being value, not view",
                return_value);
    }

    template<class PolicyType, class FunctorType, class ReturnType>
    inline std::enable_if_t<flare::is_execution_policy<PolicyType>::value &&
                            !(flare::is_view<ReturnType>::value ||
                              flare::is_reducer<ReturnType>::value ||
                              std::is_pointer<ReturnType>::value)>
    parallel_reduce(const PolicyType &policy, const FunctorType &functor,
                    ReturnType &return_value) {
        static_assert(
                !std::is_const<ReturnType>::value,
                "A const reduction result type is only allowed for a View, pointer or "
                "reducer return type!");

        detail::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
                "", policy, functor, return_value);
        detail::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
        fence(
                policy.space(),
                "flare::parallel_reduce: fence due to result being value, not view",
                return_value);
    }

    template<class FunctorType, class ReturnType>
    inline std::enable_if_t<!(flare::is_view<ReturnType>::value ||
                              flare::is_reducer<ReturnType>::value ||
                              std::is_pointer<ReturnType>::value)>
    parallel_reduce(const size_t &policy, const FunctorType &functor,
                    ReturnType &return_value) {
        static_assert(
                !std::is_const<ReturnType>::value,
                "A const reduction result type is only allowed for a View, pointer or "
                "reducer return type!");

        using policy_type =
                typename detail::ParallelReducePolicyType<void, size_t,
                        FunctorType>::policy_type;

        detail::ParallelReduceAdaptor<policy_type, FunctorType, ReturnType>::execute(
                "", policy_type(0, policy), functor, return_value);
        detail::ParallelReduceFence<typename policy_type::execution_space, ReturnType>::
        fence(
                typename policy_type::execution_space(),
                "flare::parallel_reduce: fence due to result being value, not view",
                return_value);
    }

    template<class FunctorType, class ReturnType>
    inline std::enable_if_t<!(flare::is_view<ReturnType>::value ||
                              flare::is_reducer<ReturnType>::value ||
                              std::is_pointer<ReturnType>::value)>
    parallel_reduce(const std::string &label, const size_t &policy,
                    const FunctorType &functor, ReturnType &return_value) {
        static_assert(
                !std::is_const<ReturnType>::value,
                "A const reduction result type is only allowed for a View, pointer or "
                "reducer return type!");

        using policy_type =
                typename detail::ParallelReducePolicyType<void, size_t,
                        FunctorType>::policy_type;
        detail::ParallelReduceAdaptor<policy_type, FunctorType, ReturnType>::execute(
                label, policy_type(0, policy), functor, return_value);
        detail::ParallelReduceFence<typename policy_type::execution_space, ReturnType>::
        fence(
                typename policy_type::execution_space(),
                "flare::parallel_reduce: fence due to result being value, not view",
                return_value);
    }

// ReturnValue as View or Reducer: take by copy to allow for inline construction

    template<class PolicyType, class FunctorType, class ReturnType>
    inline std::enable_if_t<flare::is_execution_policy<PolicyType>::value &&
                            (flare::is_view<ReturnType>::value ||
                             flare::is_reducer<ReturnType>::value ||
                             std::is_pointer<ReturnType>::value)>
    parallel_reduce(const std::string &label, const PolicyType &policy,
                    const FunctorType &functor, const ReturnType &return_value) {
        ReturnType return_value_impl = return_value;
        detail::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
                label, policy, functor, return_value_impl);
        detail::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
        fence(
                policy.space(),
                "flare::parallel_reduce: fence due to result being value, not view",
                return_value);
    }

    template<class PolicyType, class FunctorType, class ReturnType>
    inline std::enable_if_t<flare::is_execution_policy<PolicyType>::value &&
                            (flare::is_view<ReturnType>::value ||
                             flare::is_reducer<ReturnType>::value ||
                             std::is_pointer<ReturnType>::value)>
    parallel_reduce(const PolicyType &policy, const FunctorType &functor,
                    const ReturnType &return_value) {
        ReturnType return_value_impl = return_value;
        detail::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
                "", policy, functor, return_value_impl);
        detail::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
        fence(
                policy.space(),
                "flare::parallel_reduce: fence due to result being value, not view",
                return_value);
    }

    template<class FunctorType, class ReturnType>
    inline std::enable_if_t<flare::is_view<ReturnType>::value ||
                            flare::is_reducer<ReturnType>::value ||
                            std::is_pointer<ReturnType>::value>
    parallel_reduce(const size_t &policy, const FunctorType &functor,
                    const ReturnType &return_value) {
        using policy_type =
                typename detail::ParallelReducePolicyType<void, size_t,
                        FunctorType>::policy_type;
        ReturnType return_value_impl = return_value;
        detail::ParallelReduceAdaptor<policy_type, FunctorType, ReturnType>::execute(
                "", policy_type(0, policy), functor, return_value_impl);
        detail::ParallelReduceFence<typename policy_type::execution_space, ReturnType>::
        fence(
                typename policy_type::execution_space(),
                "flare::parallel_reduce: fence due to result being value, not view",
                return_value);
    }

    template<class FunctorType, class ReturnType>
    inline std::enable_if_t<flare::is_view<ReturnType>::value ||
                            flare::is_reducer<ReturnType>::value ||
                            std::is_pointer<ReturnType>::value>
    parallel_reduce(const std::string &label, const size_t &policy,
                    const FunctorType &functor, const ReturnType &return_value) {
        using policy_type =
                typename detail::ParallelReducePolicyType<void, size_t,
                        FunctorType>::policy_type;
        ReturnType return_value_impl = return_value;
        detail::ParallelReduceAdaptor<policy_type, FunctorType, ReturnType>::execute(
                label, policy_type(0, policy), functor, return_value_impl);
        detail::ParallelReduceFence<typename policy_type::execution_space, ReturnType>::
        fence(
                typename policy_type::execution_space(),
                "flare::parallel_reduce: fence due to result being value, not view",
                return_value);
    }

// No Return Argument

    template<class PolicyType, class FunctorType>
    inline void parallel_reduce(
            const std::string &label, const PolicyType &policy,
            const FunctorType &functor,
            std::enable_if_t<flare::is_execution_policy<PolicyType>::value> * =
            nullptr) {
        using FunctorAnalysis =
                detail::FunctorAnalysis<detail::FunctorPatternInterface::REDUCE, PolicyType,
                        FunctorType, void>;
        using value_type = std::conditional_t<(FunctorAnalysis::StaticValueSize != 0),
                typename FunctorAnalysis::value_type,
                typename FunctorAnalysis::pointer_type>;

        static_assert(
                FunctorAnalysis::has_final_member_function,
                "Calling parallel_reduce without either return value or final function.");

        using result_view_type =
                flare::View<value_type, flare::HostSpace, flare::MemoryUnmanaged>;
        result_view_type result_view;

        detail::ParallelReduceAdaptor<PolicyType, FunctorType,
                result_view_type>::execute(label, policy, functor,
                                           result_view);
    }

    template<class PolicyType, class FunctorType>
    inline void parallel_reduce(
            const PolicyType &policy, const FunctorType &functor,
            std::enable_if_t<flare::is_execution_policy<PolicyType>::value> * =
            nullptr) {
        using FunctorAnalysis =
                detail::FunctorAnalysis<detail::FunctorPatternInterface::REDUCE, PolicyType,
                        FunctorType, void>;
        using value_type = std::conditional_t<(FunctorAnalysis::StaticValueSize != 0),
                typename FunctorAnalysis::value_type,
                typename FunctorAnalysis::pointer_type>;

        static_assert(
                FunctorAnalysis::has_final_member_function,
                "Calling parallel_reduce without either return value or final function.");

        using result_view_type =
                flare::View<value_type, flare::HostSpace, flare::MemoryUnmanaged>;
        result_view_type result_view;

        detail::ParallelReduceAdaptor<PolicyType, FunctorType,
                result_view_type>::execute("", policy, functor,
                                           result_view);
    }

    template<class FunctorType>
    inline void parallel_reduce(const size_t &policy, const FunctorType &functor) {
        using policy_type =
                typename detail::ParallelReducePolicyType<void, size_t,
                        FunctorType>::policy_type;
        using FunctorAnalysis =
                detail::FunctorAnalysis<detail::FunctorPatternInterface::REDUCE, policy_type,
                        FunctorType, void>;
        using value_type = std::conditional_t<(FunctorAnalysis::StaticValueSize != 0),
                typename FunctorAnalysis::value_type,
                typename FunctorAnalysis::pointer_type>;

        static_assert(
                FunctorAnalysis::has_final_member_function,
                "Calling parallel_reduce without either return value or final function.");

        using result_view_type =
                flare::View<value_type, flare::HostSpace, flare::MemoryUnmanaged>;
        result_view_type result_view;

        detail::ParallelReduceAdaptor<policy_type, FunctorType,
                result_view_type>::execute("",
                                           policy_type(0, policy),
                                           functor, result_view);
    }

    template<class FunctorType>
    inline void parallel_reduce(const std::string &label, const size_t &policy,
                                const FunctorType &functor) {
        using policy_type =
                typename detail::ParallelReducePolicyType<void, size_t,
                        FunctorType>::policy_type;
        using FunctorAnalysis =
                detail::FunctorAnalysis<detail::FunctorPatternInterface::REDUCE, policy_type,
                        FunctorType, void>;
        using value_type = std::conditional_t<(FunctorAnalysis::StaticValueSize != 0),
                typename FunctorAnalysis::value_type,
                typename FunctorAnalysis::pointer_type>;

        static_assert(
                FunctorAnalysis::has_final_member_function,
                "Calling parallel_reduce without either return value or final function.");

        using result_view_type =
                flare::View<value_type, flare::HostSpace, flare::MemoryUnmanaged>;
        result_view_type result_view;

        detail::ParallelReduceAdaptor<policy_type, FunctorType,
                result_view_type>::execute(label,
                                           policy_type(0, policy),
                                           functor, result_view);
    }

}  // namespace flare

#endif  // FLARE_CORE_PARALLEL_PARALLEL_REDUCE_H_
