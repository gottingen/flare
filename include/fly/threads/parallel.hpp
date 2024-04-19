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

#include "async_queue.hpp"

#include <numeric>
#include <array>
#include <vector>
#include <cmath>

using dim_t = std::array<size_t, 4>;

#define NTHREADS 8
static std::vector<threads::async_queue> queues(NTHREADS);

template<typename FUNC, size_t DIM>
struct work;

class parallel_mat
{
    const void * const func;  //void * makes me sad :(
    const dim_t &bound;

public:
    template<typename FUNC>
    parallel_mat(const dim_t &iterations, FUNC func)
        : func(static_cast<void*>(&func))
        , bound(iterations)
    {
        using std::array;
        using std::begin;
        using std::end;
        using std::partial_sum;

        array<size_t, 4> w = {{0, 0, 0, 0}};
        func(w);
        //array<size_t, 4> nelems;
        //partial_sum(begin(bound), end(bound), begin(nelems));
        work<FUNC, 3> ww;
        ww(this, w);
    }
    const dim_t& getBound() const {return bound;}
    const void* getFunc() const {return func;}
};

template<typename FUNC, size_t DIM>
struct work {
    void operator()(const parallel_mat *ref, dim_t iterations)
    {
        using std::get;

        const size_t &b    = get<DIM>(ref->getBound());
              size_t &iter = get<DIM>(iterations);
        if(DIM==3) {
            work<FUNC, DIM-1> w;
            for(;  iter < b; iter++) {
                queues[iter % NTHREADS].enqueue( w, ref, iterations);
            }
            for(auto &q : queues) q.sync();
        }
        else {
            work<FUNC, DIM-1> w;
            for(;  iter < b; iter++) {
                w(ref, iterations);
            }
        }
    }
};

template<typename FUNC>
struct work<FUNC, 0> {
    void operator()(const parallel_mat *ref, dim_t iterations)
    {
        using std::get;

        const size_t &b    = get<0>(ref->getBound());
              size_t &iter = get<0>(iterations);
        const auto f = *(static_cast<const FUNC * const>(ref->getFunc()));
        for(;  iter < b; iter++) {
            f(iterations);
        }
    }
};
