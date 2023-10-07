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
//
// Created by jeff on 23-10-6.
//

#ifndef FLARE_KERNEL_COMMON_PREDICATES_H_
#define FLARE_KERNEL_COMMON_PREDICATES_H_

#include <flare/core/numeric_traits.h>

namespace flare {

    /**
     * @brief Struct template for a greater-than predicate
     * @tparam T Type to be compared
     */
    template<typename T>
    struct GT {
        using value_type = T;
        static_assert(!flare::ArithTraits<T>::is_complex,
                      "Please define custom predicates for ordering complex types");

        /**
         * @brief Return true if a is greater than b
         * @param a First value to be compared
         * @param b Second value to be compared
         */
        FLARE_INLINE_FUNCTION constexpr bool operator()(const value_type &a,
                                                        const value_type &b) const
        noexcept {
            return a > b;
        }
    };

    /*! \brief "Greater-than-or-equal" predicate, a >= b
        \tparam T the type to compare
    */
    template<typename T>
    struct GTE {
        using value_type = T;
        static_assert(!flare::ArithTraits<T>::is_complex,
                      "Please define custom predicates for ordering complex types");

        /// \brief return a >= b
        FLARE_INLINE_FUNCTION constexpr bool operator()(const value_type &a,
                                                        const value_type &b) const
        noexcept {
            return a >= b;
        }
    };

    /*! \brief "Less-than" predicate, a < b
        \tparam T the type to compare
    */
    template<typename T>
    struct LT {
        using value_type = T;
        static_assert(!flare::ArithTraits<T>::is_complex,
                      "Please define custom predicates for ordering complex types");

        /// \brief return a < b
        FLARE_INLINE_FUNCTION constexpr bool operator()(const value_type &a,
                                                        const value_type &b) const noexcept {
            return a < b;
        }
    };

    /*! \brief "Less-than-or-equal" predicate, a <= b
        \tparam T the type to compare
    */
    template<typename T>
    struct LTE {
        using value_type = T;
        static_assert(!flare::ArithTraits<T>::is_complex,
                      "Please define custom predicates for ordering complex types");

        /// \brief return a <= b
        FLARE_INLINE_FUNCTION constexpr bool operator()(const value_type &a,
                                                        const value_type &b) const
        noexcept {
            return a <= b;
        }
    };

    /*! \brief "Equal" predicate, a == b
        \tparam T the type to compare
    */
    template<typename T>
    struct Equal {
        using value_type = T;

        /// \brief return a == b
        FLARE_INLINE_FUNCTION constexpr bool operator()(const value_type &a,
                                                        const value_type &b) const {
            return a == b;
        }
    };

    /**
     * @brief Struct template for inverting a predicate
     * @tparam Pred Predicate type to be inverted
     */
    template<typename Pred>
    struct Neg {
        using value_type = typename Pred::value_type;

        /**
         * @brief Constructor
         * @param pred Predicate object to be inverted
         */
        FLARE_INLINE_FUNCTION
        constexpr Neg(const Pred &pred) : pred_(pred) {}

        /**
         * @brief Return the boolean inverse of the underlying predicate
         * @param a First value to be compared by the predicate
         * @param b Second value to be compared by the predicate
         * @return Boolean inverse of the result of the predicate applied to a and b
         */
        FLARE_INLINE_FUNCTION constexpr bool operator()(const value_type &a,
                                                        const value_type &b) const {
            return !pred_(a, b);
        }

    private:
        Pred pred_;  //< Underlying predicate object
    };

    /*! \brief Reflect a predicate, pred(b, a)
        \tparam Pred the type of the predicate to reflect
    */
    template<typename Pred>
    struct Refl {
        using value_type = typename Pred::value_type;

        FLARE_INLINE_FUNCTION
        constexpr Refl(const Pred &pred) : pred_(pred) {}

        /// \brief return the underlying binary predicate with reversed arguments
        FLARE_INLINE_FUNCTION constexpr bool operator()(const value_type &a,
                                                        const value_type &b) const {
            return pred_(b, a);
        }

    private:
        Pred pred_;
    };

}  // namespace flare

#endif  // FLARE_KERNEL_COMMON_PREDICATES_H_
