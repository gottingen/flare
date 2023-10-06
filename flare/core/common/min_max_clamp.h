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

#ifndef FLARE_CORE_COMMON_MIN_MAX_CLAMP_H_
#define FLARE_CORE_COMMON_MIN_MAX_CLAMP_H_

#include <flare/core/defines.h>
#include <flare/core/pair.h>

#include <initializer_list>

namespace flare {

    // clamp
    template<class T>
    constexpr FLARE_INLINE_FUNCTION const T &clamp(const T &value, const T &lo,
                                                   const T &hi) {
        FLARE_EXPECTS(!(hi < lo));
        return (value < lo) ? lo : (hi < value) ? hi : value;
    }

    template<class T, class ComparatorType>
    constexpr FLARE_INLINE_FUNCTION const T &clamp(const T &value, const T &lo,
                                                   const T &hi,
                                                   ComparatorType comp) {
        FLARE_EXPECTS(!comp(hi, lo));
        return comp(value, lo) ? lo : comp(hi, value) ? hi : value;
    }

    // max
    template<class T>
    constexpr FLARE_INLINE_FUNCTION const T &max(const T &a, const T &b) {
        return (a < b) ? b : a;
    }

    template<class T, class ComparatorType>
    constexpr FLARE_INLINE_FUNCTION const T &max(const T &a, const T &b,
                                                 ComparatorType comp) {
        return comp(a, b) ? b : a;
    }

    template<class T>
    FLARE_INLINE_FUNCTION constexpr T max(std::initializer_list<T> ilist) {
        auto first = ilist.begin();
        auto const last = ilist.end();
        auto result = *first;
        if (first == last) return result;
        while (++first != last) {
            if (result < *first) result = *first;
        }
        return result;
    }

    template<class T, class Compare>
    FLARE_INLINE_FUNCTION constexpr T max(std::initializer_list<T> ilist,
                                          Compare comp) {
        auto first = ilist.begin();
        auto const last = ilist.end();
        auto result = *first;
        if (first == last) return result;
        while (++first != last) {
            if (comp(result, *first)) result = *first;
        }
        return result;
    }

    // min
    template<class T>
    constexpr FLARE_INLINE_FUNCTION const T &min(const T &a, const T &b) {
        return (b < a) ? b : a;
    }

    template<class T, class ComparatorType>
    constexpr FLARE_INLINE_FUNCTION const T &min(const T &a, const T &b,
                                                 ComparatorType comp) {
        return comp(b, a) ? b : a;
    }

    template<class T>
    FLARE_INLINE_FUNCTION constexpr T min(std::initializer_list<T> ilist) {
        auto first = ilist.begin();
        auto const last = ilist.end();
        auto result = *first;
        if (first == last) return result;
        while (++first != last) {
            if (*first < result) result = *first;
        }
        return result;
    }

    template<class T, class Compare>
    FLARE_INLINE_FUNCTION constexpr T min(std::initializer_list<T> ilist,
                                          Compare comp) {
        auto first = ilist.begin();
        auto const last = ilist.end();
        auto result = *first;
        if (first == last) return result;
        while (++first != last) {
            if (comp(*first, result)) result = *first;
        }
        return result;
    }

// minmax
    template<class T>
    constexpr FLARE_INLINE_FUNCTION auto minmax(const T &a, const T &b) {
        using return_t = ::flare::pair<const T &, const T &>;
        return (b < a) ? return_t{b, a} : return_t{a, b};
    }

    template<class T, class ComparatorType>
    constexpr FLARE_INLINE_FUNCTION auto minmax(const T &a, const T &b,
                                                ComparatorType comp) {
        using return_t = ::flare::pair<const T &, const T &>;
        return comp(b, a) ? return_t{b, a} : return_t{a, b};
    }

    template<class T>
    FLARE_INLINE_FUNCTION constexpr flare::pair<T, T> minmax(
            std::initializer_list<T> ilist) {
        auto first = ilist.begin();
        auto const last = ilist.end();
        auto next = first;
        flare::pair<T, T> result{*first, *first};
        if (first == last || ++next == last) return result;
        if (*next < *first)
            result.first = *next;
        else
            result.second = *next;
        first = next;
        while (++first != last) {
            if (++next == last) {
                if (*first < result.first)
                    result.first = *first;
                else if (!(*first < result.second))
                    result.second = *first;
                break;
            }
            if (*next < *first) {
                if (*next < result.first) result.first = *next;
                if (!(*first < result.second)) result.second = *first;
            } else {
                if (*first < result.first) result.first = *first;
                if (!(*next < result.second)) result.second = *next;
            }
            first = next;
        }
        return result;
    }

    template<class T, class Compare>
    FLARE_INLINE_FUNCTION constexpr flare::pair<T, T> minmax(
            std::initializer_list<T> ilist, Compare comp) {
        auto first = ilist.begin();
        auto const last = ilist.end();
        auto next = first;
        flare::pair<T, T> result{*first, *first};
        if (first == last || ++next == last) return result;
        if (comp(*next, *first))
            result.first = *next;
        else
            result.second = *next;
        first = next;
        while (++first != last) {
            if (++next == last) {
                if (comp(*first, result.first))
                    result.first = *first;
                else if (!comp(*first, result.second))
                    result.second = *first;
                break;
            }
            if (comp(*next, *first)) {
                if (comp(*next, result.first)) result.first = *next;
                if (!comp(*first, result.second)) result.second = *first;
            } else {
                if (comp(*first, result.first)) result.first = *first;
                if (!comp(*next, result.second)) result.second = *next;
            }
            first = next;
        }
        return result;
    }


}  // namespace flare

#endif  // FLARE_CORE_COMMON_MIN_MAX_CLAMP_H_
