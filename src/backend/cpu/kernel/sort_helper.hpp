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

#include <err_cpu.hpp>
#include <tuple>

namespace flare {
namespace cpu {
namespace kernel {
template<typename Tk, typename Tv>
using IndexPair = std::tuple<Tk, Tv>;

template<typename Tk, typename Tv, bool isAscending>
struct IPCompare {
    bool operator()(const IndexPair<Tk, Tv> &lhs,
                    const IndexPair<Tk, Tv> &rhs) {
        // Check stable sort condition
        Tk lhsVal = std::get<0>(lhs);
        Tk rhsVal = std::get<0>(rhs);
        if (isAscending)
            return (lhsVal < rhsVal);
        else
            return (lhsVal > rhsVal);
    }
};

template<typename Tk, typename Tv>
using KeyIndexPair = std::tuple<Tk, Tv, uint>;

template<typename Tk, typename Tv, bool isAscending>
struct KIPCompareV {
    bool operator()(const KeyIndexPair<Tk, Tv> &lhs,
                    const KeyIndexPair<Tk, Tv> &rhs) {
        // Check stable sort condition
        Tk lhsVal = std::get<0>(lhs);
        Tk rhsVal = std::get<0>(rhs);
        if (isAscending)
            return (lhsVal < rhsVal);
        else
            return (lhsVal > rhsVal);
    }
};

template<typename Tk, typename Tv, bool isAscending>
struct KIPCompareK {
    bool operator()(const KeyIndexPair<Tk, Tv> &lhs,
                    const KeyIndexPair<Tk, Tv> &rhs) {
        uint lhsVal = std::get<2>(lhs);
        uint rhsVal = std::get<2>(rhs);
        if (isAscending)
            return (lhsVal < rhsVal);
        else
            return (lhsVal > rhsVal);
    }
};
}  // namespace kernel
}  // namespace cpu
}  // namespace flare
