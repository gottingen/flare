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

#ifndef SIMD_OPS_H_
#define SIMD_OPS_H_

#include <flare/simd/simd.h>

class plus {
public:
    template<class T>
    auto on_host(T const &a, T const &b) const {
        return a + b;
    }

    template<class T>
    FLARE_INLINE_FUNCTION auto on_device(T const &a, T const &b) const {
        return a + b;
    }
};

class minus {
public:
    template<class T>
    auto on_host(T const &a, T const &b) const {
        return a - b;
    }

    template<class T>
    FLARE_INLINE_FUNCTION auto on_device(T const &a, T const &b) const {
        return a - b;
    }
};

class multiplies {
public:
    template<class T>
    auto on_host(T const &a, T const &b) const {
        return a * b;
    }

    template<class T>
    FLARE_INLINE_FUNCTION auto on_device(T const &a, T const &b) const {
        return a * b;
    }
};

class divides {
public:
    template<class T>
    auto on_host(T const &a, T const &b) const {
        return a / b;
    }

    template<class T>
    FLARE_INLINE_FUNCTION auto on_device(T const &a, T const &b) const {
        return a / b;
    }
};

class absolutes {
    template<typename T>
    static FLARE_FUNCTION auto abs_impl(T const &x) {
        if constexpr (std::is_signed_v<T>) {
            return flare::abs(x);
        }
        return x;
    }

public:
    template<typename T>
    auto on_host(T const &a) const {
        return flare::experimental::abs(a);
    }

    template<typename T>
    auto on_host_serial(T const &a) const {
        return abs_impl(a);
    }

    template<typename T>
    FLARE_INLINE_FUNCTION auto on_device(T const &a) const {
        return flare::experimental::abs(a);
    }

    template<typename T>
    FLARE_INLINE_FUNCTION auto on_device_serial(T const &a) const {
        return abs_impl(a);
    }
};

class shift_right {
public:
    template<typename T, typename U>
    auto on_host(T &&a, U &&b) const {
        return a >> b;
    }

    template<typename T, typename U>
    FLARE_INLINE_FUNCTION auto on_device(T &&a, U &&b) const {
        return a >> b;
    }
};

class shift_left {
public:
    template<typename T, typename U>
    auto on_host(T &&a, U &&b) const {
        return a << b;
    }

    template<typename T, typename U>
    FLARE_INLINE_FUNCTION auto on_device(T &&a, U &&b) const {
        return a << b;
    }
};

#endif  // SIMD_OPS_H_
