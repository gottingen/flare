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

#include <fly/array.h>
#include <fly/data.h>
#include <fly/seq.h>

#include "error.hpp"

#include <cmath>

namespace fly {
int end = -1;
seq span(fly_span);

void seq::init(double begin, double end, double step) {
    this->s.begin = begin;
    this->s.end   = end;
    this->s.step  = step;
    if (step != 0) {  // Not Span
        size = std::fabs((end - begin) / step) + 1;
    } else {
        size = 0;
    }
}

#ifndef signbit
// wtf windows?!
inline int signbit(double x) {
    if (x < 0) { return -1; }
    return 0;
}
#endif

seq::~seq() = default;

seq::seq(double length) : s{}, size{}, m_gfor(false) {
    if (length < 0) {
        init(0, length, 1);
    } else {
        init(0, length - 1, 1);
    }
}

seq::seq(const fly_seq& s_) : s{}, size{}, m_gfor(false) {
    init(s_.begin, s_.end, s_.step);
}

seq& seq::operator=(const fly_seq& s_) {
    init(s_.begin, s_.end, s_.step);
    return *this;
}

seq::seq(double begin, double end, double step) : s{}, size{}, m_gfor(false) {
    if (step == 0) {
        if (begin != end) {  // Span
            FLY_THROW_ERR("Invalid step size", FLY_ERR_ARG);
        }
    }
    if ((signbit(end) == signbit(begin)) &&
        (signbit(end - begin) != signbit(step))) {
        FLY_THROW_ERR("Sequence is invalid", FLY_ERR_ARG);
    }
    init(begin, end, step);
}

seq::seq(seq other,  // NOLINT(performance-unnecessary-value-param)
         bool is_gfor)
    : s(other.s), size(other.size), m_gfor(is_gfor) {}

seq::operator array() const {
    double diff = s.end - s.begin;
    dim_t len   = static_cast<int>(
        (diff + std::fabs(s.step) * (signbit(diff) == 0 ? 1 : -1)) / s.step);

    array tmp = (m_gfor) ? range(1, 1, 1, len, 3) : range(len);

    array res = s.begin + s.step * tmp;
    return res;
}
}  // namespace fly
