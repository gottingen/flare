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

#include <types.hpp>

#include <common/half.hpp>
#include <common/util.hpp>
#include <type_util.hpp>

#include <cmath>
#include <sstream>
#include <string>

using flare::common::half;
using flare::common::toString;

using std::isinf;
using std::stringstream;

namespace flare {
namespace opencl {

template<typename T>
inline std::string ToNumStr<T>::operator()(T val) {
    ToNum<T> toNum;
    return toString(toNum(val));
}

template<>
std::string ToNumStr<float>::operator()(float val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (isinf(val)) { return val < 0.f ? NINF : PINF; }
    return toString(val);
}

template<>
std::string ToNumStr<double>::operator()(double val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (isinf(val)) { return val < 0. ? NINF : PINF; }
    return toString(val);
}

template<>
std::string ToNumStr<cfloat>::operator()(cfloat val) {
    ToNumStr<float> realStr;
    stringstream s;
    s << "{" << realStr(val.s[0]) << "," << realStr(val.s[1]) << "}";
    return s.str();
}

template<>
std::string ToNumStr<cdouble>::operator()(cdouble val) {
    ToNumStr<double> realStr;
    stringstream s;
    s << "{" << realStr(val.s[0]) << "," << realStr(val.s[1]) << "}";
    return s.str();
}

template<>
std::string ToNumStr<half>::operator()(half val) {
    using namespace std;
    using namespace common;
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (isinf(val)) { return val < 0.f ? NINF : PINF; }
    return toString(val);
}

template<>
template<>
std::string ToNumStr<half>::operator()<float>(float val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (isinf(half(val))) { return val < 0.f ? NINF : PINF; }
    return toString(val);
}

#define INSTANTIATE(TYPE) template struct ToNumStr<TYPE>

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(cfloat);
INSTANTIATE(cdouble);
INSTANTIATE(short);
INSTANTIATE(ushort);
INSTANTIATE(int);
INSTANTIATE(uint);
INSTANTIATE(intl);
INSTANTIATE(uintl);
INSTANTIATE(uchar);
INSTANTIATE(char);
INSTANTIATE(half);

#undef INSTANTIATE

}  // namespace opencl
}  // namespace flare
