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

#include <flare/core/common/string_manipulation.h>
#include <climits>

namespace {

    FLARE_FUNCTION constexpr bool test_strlen() {
        using flare::detail::strlen;
        constexpr char str[] = "How many characters does this string contain?";
        static_assert(strlen(str) == 45);  // without null character
        static_assert(sizeof str == 46);   // with null character
        static_assert(strlen("") == 0);
        return true;
    }

    static_assert(test_strlen());

    FLARE_FUNCTION constexpr bool test_strcmp() {
        using flare::detail::strcmp;
        constexpr char cat1[] = "Heathcliff";
        constexpr char cat2[] = "Snagglepuss";
        constexpr char cat3[] = "Hobbes";
        constexpr char cat4[] = "Garfield";
        static_assert(strcmp(cat1, cat1) == 0);
#if (!defined(FLARE_COMPILER_NVCC) || \
     ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 3))) && \
    (!defined(__INTEL_COMPILER_BUILD_DATE) || \
     (__INTEL_COMPILER_BUILD_DATE >= 20210228))
        static_assert(strcmp(cat1, cat2) < 0);
        static_assert(strcmp(cat1, cat3) < 0);
#endif
        static_assert(strcmp(cat1, cat4) > 0);
        static_assert(strcmp(cat2, cat2) == 0);
        static_assert(strcmp(cat2, cat3) > 0);
        static_assert(strcmp(cat2, cat4) > 0);
        static_assert(strcmp(cat3, cat3) == 0);
        static_assert(strcmp(cat3, cat4) > 0);
        static_assert(strcmp(cat4, cat4) == 0);
        return true;
    }

    static_assert(test_strcmp());

    FLARE_FUNCTION constexpr bool test_strncmp() {
        using flare::detail::strncmp;
        constexpr char greet1[] = "Hello, world!";
        constexpr char greet2[] = "Hello, everybody!";
        constexpr char greet3[] = "Hello, somebody!";
        static_assert(strncmp(greet1, greet2, 13) > 0);
        static_assert(strncmp(greet2, greet1, 13) < 0);
        static_assert(strncmp(greet2, greet1, 7) == 0);
        static_assert(strncmp(greet2 + 12, greet3 + 11, 5) == 0);
        static_assert(strncmp(greet1, greet2, 0) == 0);
        return true;
    }

    static_assert(test_strncmp());

    FLARE_FUNCTION constexpr bool strcpy_helper(const char *dest, const char *src,
                                                const char *ref) {
        using flare::detail::strcmp;
        using flare::detail::strcpy;
        char buffer[50] = {};
        strcpy(buffer, dest);
        strcpy(buffer, src);
        return strcmp(buffer, ref) == 0;
    }

    FLARE_FUNCTION constexpr bool test_strcpy() {
        static_assert(strcpy_helper("abcdef", "hi", "hi\0\0\0f"));
        return true;
    }

    static_assert(test_strcpy());

    FLARE_FUNCTION constexpr bool strncpy_helper(const char *dest, const char *src,
                                                 std::size_t count,
                                                 const char *ref) {
        using flare::detail::strcmp;
        using flare::detail::strlen;
        using flare::detail::strncpy;
        char buffer[50] = {};
        strncpy(buffer, dest, strlen(dest));
        strncpy(buffer, src, count);
        return strcmp(buffer, ref) == 0;
    }

    FLARE_FUNCTION constexpr bool test_strncpy() {
        static_assert(strncpy_helper("abcdef", "hi", 5, "hi\0\0\0f"));
        static_assert(strncpy_helper("abcdef", "hi", 0, "abcdef"));
        return true;
    }

    static_assert(test_strncpy());

    FLARE_FUNCTION constexpr bool strcat_helper(const char *dest, const char *src,
                                                const char *ref) {
        using flare::detail::strcat;
        using flare::detail::strcmp;
        char buffer[50] = {};
        strcat(buffer, dest);
        strcat(buffer, src);
        return strcmp(buffer, ref) == 0;
    }

    FLARE_FUNCTION constexpr bool test_strcat() {
        static_assert(strcat_helper("Hello ", "World!", "Hello World!"));
        static_assert(strcat_helper("Hello World!", " Goodbye World!",
                                    "Hello World! Goodbye World!"));
        return true;
    }

    static_assert(test_strcat());

    FLARE_FUNCTION constexpr bool strncat_helper(const char *dest, const char *src,
                                                 std::size_t count,
                                                 const char *ref) {
        using flare::detail::strcmp;
        using flare::detail::strlen;
        using flare::detail::strncat;
        char buffer[50] = {};
        strncat(buffer, dest, strlen(dest));
        strncat(buffer, src, count);
        return strcmp(buffer, ref) == 0;
    }

    FLARE_FUNCTION constexpr bool test_strncat() {
        static_assert(
                strncat_helper("Hello World!", " Goodbye World!", 3, "Hello World! Go"));
        static_assert(
                strncat_helper("Hello World!", " Goodbye World!", 0, "Hello World!"));
        return true;
    }

    static_assert(test_strncat());

    template<class Integral>
    FLARE_FUNCTION constexpr bool to_chars_helper(Integral val, char const *ref) {
        using flare::detail::strcmp;
        using flare::detail::strlen;
        using flare::detail::to_chars_i;
        constexpr int BUFFER_SIZE = 21;
        char buffer[BUFFER_SIZE] = {};
        return (buffer + strlen(ref) ==
                to_chars_i(buffer, buffer + BUFFER_SIZE, val).ptr) &&
               (strcmp(buffer, ref) == 0);
    }

    FLARE_FUNCTION constexpr bool test_to_chars() {
        static_assert(to_chars_helper(0, "0"));
        static_assert(to_chars_helper(123, "123"));
        static_assert(to_chars_helper(-456, "-456"));
        static_assert(to_chars_helper(INT_MAX, "2147483647"));
        static_assert(to_chars_helper(INT_MIN, "-2147483648"));

        static_assert(to_chars_helper(0u, "0"));
        static_assert(to_chars_helper(78u, "78"));
        static_assert(to_chars_helper(UINT_MAX, "4294967295"));

        static_assert(to_chars_helper(0ll, "0"));
        static_assert(to_chars_helper(LLONG_MAX, "9223372036854775807"));
        static_assert(to_chars_helper(LLONG_MIN, "-9223372036854775808"));

        static_assert(to_chars_helper(0ull, "0"));
        static_assert(to_chars_helper(ULLONG_MAX, "18446744073709551615"));

        return true;
    }

    static_assert(test_to_chars());

}  // namespace
