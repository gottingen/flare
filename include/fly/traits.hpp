// Copyright 2023 The EA Authors.
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

#pragma once

#ifdef __cplusplus

#include <complex>
#include <fly/defines.h>
#include <fly/complex.h>
#include <fly/half.h>

namespace fly {

template<typename T> struct dtype_traits;

template<>
struct dtype_traits<float> {
    enum {
        fly_type = f32,
        ctype = f32
    };
    typedef float base_type;
    static const char* getName() { return "float"; }
};

template<>
struct dtype_traits<fly::cfloat> {
    enum {
        fly_type = c32 ,
        ctype = c32
    };
    typedef float base_type;
    static const char* getName() { return "std::complex<float>"; }
};

template<>
struct dtype_traits<std::complex<float> > {
    enum {
        fly_type = c32 ,
        ctype = c32
    };
    typedef float base_type;
    static const char* getName() { return "std::complex<float>"; }
};

template<>
struct dtype_traits<double> {
    enum {
        fly_type = f64 ,
        ctype = f32
    };
    typedef double base_type;
    static const char* getName() { return "double"; }
};

template<>
struct dtype_traits<fly::cdouble> {
    enum {
        fly_type = c64 ,
        ctype = c64
    };
    typedef double base_type;
    static const char* getName() { return "std::complex<double>"; }
};

template<>
struct dtype_traits<std::complex<double> > {
    enum {
        fly_type = c64 ,
        ctype = c64
    };
    typedef double base_type;
    static const char* getName() { return "std::complex<double>"; }
};

template<>
struct dtype_traits<char> {
    enum {
        fly_type = b8 ,
        ctype = f32
    };
    typedef char base_type;
    static const char* getName() { return "char"; }
};

template<>
struct dtype_traits<int> {
    enum {
        fly_type = s32 ,
        ctype = f32
    };
    typedef int base_type;
    static const char* getName() { return "int"; }
};

template<>
struct dtype_traits<unsigned> {
    enum {
        fly_type = u32 ,
        ctype = f32
    };
    typedef unsigned base_type;
    static const char* getName() { return "uint"; }
};

template<>
struct dtype_traits<unsigned char> {
    enum {
        fly_type = u8 ,
        ctype = f32
    };
    typedef unsigned char base_type;
    static const char* getName() { return "uchar"; }
};

template<>
struct dtype_traits<long long> {
    enum {
        fly_type = s64 ,
        ctype = s64
    };
    typedef long long base_type;
    static const char* getName() { return "long"; }
};

template<>
struct dtype_traits<unsigned long long> {
    enum {
        fly_type = u64 ,
        ctype = u64
    };
    typedef unsigned long long base_type;
    static const char* getName() { return "ulong"; }
};


template<>
struct dtype_traits<short> {
    enum {
        fly_type = s16 ,
        ctype = s16
    };
    typedef short base_type;
    static const char* getName() { return "short"; }
};

template<>
struct dtype_traits<unsigned short> {
    enum {
        fly_type = u16 ,
        ctype = u16
    };
    typedef unsigned short base_type;
    static const char* getName() { return "ushort"; }
};


template<>
struct dtype_traits<half> {
    enum {
        fly_type = f16 ,
        ctype = f16
    };
    typedef half base_type;
    static const char* getName() { return "half"; }
};

}

#endif
