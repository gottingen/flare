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

#ifndef FLARE_CORE_COMMON_STRING_MANIPULATION_H_
#define FLARE_CORE_COMMON_STRING_MANIPULATION_H_

#include <flare/core/defines.h>
#include <cstddef>
#include <type_traits>

namespace flare::detail {

    // This header provides a subset of the functionality from <cstring>.  In
    // contrast to the standard library header, functions are usable on the device
    // and in constant expressions.  It also includes functionality from <charconv>
    // to convert an integer value to a character sequence.

    // returns the length of a given string
    FLARE_INLINE_FUNCTION constexpr std::size_t strlen(const char *str) {
        std::size_t i = 0;
        while (str[i] != '\0') {
            ++i;
        }
        return i;
    }

    // compares two strings
    FLARE_INLINE_FUNCTION constexpr int strcmp(const char *lhs, const char *rhs) {
        while (*lhs == *rhs++) {
            if (*lhs++ == '\0') {
                return 0;
            }
        }
        return static_cast<unsigned int>(*lhs) -
               static_cast<unsigned int>(*(rhs - 1));
    }

    // compares a certain number of characters from two strings
    FLARE_INLINE_FUNCTION constexpr int strncmp(const char *lhs, const char *rhs,
                                                std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            if (lhs[i] != rhs[i]) {
                return lhs[i] < rhs[i] ? -1 : 1;
            } else if (lhs[i] == '\0') {
                return 0;
            }
        }
        return 0;
    }

    // copies one string to another
    FLARE_INLINE_FUNCTION constexpr char *strcpy(char *dest, const char *src) {
        char *d = dest;
        for (; (*d = *src) != '\0'; ++d, ++src) {
        }
        return dest;
    }

    // copies a certain amount of characters from one string to another
    FLARE_INLINE_FUNCTION constexpr char *strncpy(char *dest, const char *src,
                                                  std::size_t count) {
        if (count != 0) {
            char *d = dest;
            do {
                if ((*d++ = *src++) == '\0') {
                    while (--count != 0) {
                        *d++ = '\0';
                    }
                    break;
                }
            } while (--count != 0);
        }
        return dest;
    }

    // concatenates two strings
    FLARE_INLINE_FUNCTION constexpr char *strcat(char *dest, const char *src) {
        char *d = dest;
        for (; *d != '\0'; ++d) {
        }
        while ((*d++ = *src++) != '\0') {
        }
        return dest;
    }

    // concatenates a certain amount of characters of two strings
    FLARE_INLINE_FUNCTION constexpr char *strncat(char *dest, const char *src,
                                                  std::size_t count) {
        if (count != 0) {
            char *d = dest;
            for (; *d != '\0'; ++d) {
            }
            do {
                if ((*d = *src++) == '\0') {
                    break;
                }
                d++;
            } while (--count != 0);
            *d = '\0';
        }
        return dest;
    }

    template<class Unsigned>
    FLARE_FUNCTION constexpr unsigned int to_chars_len(Unsigned val) {
        unsigned int const base = 10;
        static_assert(std::is_integral<Unsigned>::value, "implementation bug");
        static_assert(std::is_unsigned<Unsigned>::value, "implementation bug");
        unsigned int n = 1;
        while (val >= base) {
            val /= base;
            ++n;
        }
        return n;
    }

    template<class Unsigned>
    FLARE_FUNCTION constexpr void to_chars_impl(char *first, unsigned int len,
                                                Unsigned val) {
        unsigned int const base = 10;
        static_assert(std::is_integral<Unsigned>::value, "implementation bug");
        static_assert(std::is_unsigned<Unsigned>::value, "implementation bug");
        unsigned int pos = len - 1;
        while (val > 0) {
            auto const num = val % base;
            val /= base;
            first[pos] = '0' + num;
            --pos;
        }
    }

    // define values of portable error conditions that correspond to the POSIX error
    // codes
    enum class errc {
        value_too_large = 75  // equivalent POSIX error is EOVERFLOW
    };
    struct to_chars_result {
        char *ptr;
        errc ec;
    };

    // converts an integer value to a character sequence
    template<class Integral>
    FLARE_FUNCTION constexpr to_chars_result to_chars_i(char *first, char *last,
                                                        Integral value) {
        using Unsigned = std::conditional_t<sizeof(Integral) <= sizeof(unsigned int),
                unsigned int, unsigned long long>;
        Unsigned unsigned_val = value;
        if (value == 0) {
            *first = '0';
            return {first + 1, {}};
        } else if constexpr (std::is_signed<Integral>::value) {
            if (value < 0) {
                *first++ = '-';
                unsigned_val = Unsigned(~value) + Unsigned(1);
            }
        }
        std::ptrdiff_t const len = to_chars_len(unsigned_val);
        if (last - first < len) {
            return {last, errc::value_too_large};
        }
        to_chars_impl(first, len, unsigned_val);
        return {first + len, {}};
    }

}  // namespace flare::detail

#endif  // FLARE_CORE_COMMON_STRING_MANIPULATION_H_
