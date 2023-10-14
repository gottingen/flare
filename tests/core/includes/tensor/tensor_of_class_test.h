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

#include <doctest.h>

#include <flare/core.h>
#include <sstream>
#include <iostream>

namespace Test {

    template<class Space>
    struct NesteDTensor {
        flare::Tensor<int *, Space> member;

    public:
        FLARE_INLINE_FUNCTION
        NesteDTensor() : member() {}

        FLARE_INLINE_FUNCTION
        NesteDTensor &operator=(const flare::Tensor<int *, Space> &lhs) {
            member = lhs;
            if (member.extent(0)) flare::atomic_add(&member(0), 1);
            return *this;
        }

        FLARE_INLINE_FUNCTION
        ~NesteDTensor() {
            if (member.extent(0)) {
                flare::atomic_add(&member(0), -1);
            }
        }
    };

    template<class Space>
    struct NesteDTensorFunctor {
        flare::Tensor<NesteDTensor<Space> *, Space> nested;
        flare::Tensor<int *, Space> array;

        NesteDTensorFunctor(const flare::Tensor<NesteDTensor<Space> *, Space> &arg_nested,
                          const flare::Tensor<int *, Space> &arg_array)
                : nested(arg_nested), array(arg_array) {}

        FLARE_INLINE_FUNCTION
        void operator()(int i) const { nested[i] = array; }
    };

    template<class Space>
    void tensor_nested_tensor() {
        flare::Tensor<int *, Space> tracking("tracking", 1);

        typename flare::Tensor<int *, Space>::HostMirror host_tracking =
                flare::create_mirror(tracking);

        {
            flare::Tensor<NesteDTensor<Space> *, Space> a("a_nested_tensor", 2);

            flare::parallel_for(flare::RangePolicy<Space>(0, 2),
                                NesteDTensorFunctor<Space>(a, tracking));
            flare::deep_copy(host_tracking, tracking);
            REQUIRE_EQ(2, host_tracking(0));

            flare::Tensor<NesteDTensor<Space> *, Space> b("b_nested_tensor", 2);
            flare::parallel_for(flare::RangePolicy<Space>(0, 2),
                                NesteDTensorFunctor<Space>(b, tracking));
            flare::deep_copy(host_tracking, tracking);
            REQUIRE_EQ(4, host_tracking(0));
        }

        flare::deep_copy(host_tracking, tracking);

        REQUIRE_EQ(0, host_tracking(0));
    }

    TEST_CASE("TEST_CATEGORY, tensor_nested_tensor") { tensor_nested_tensor<TEST_EXECSPACE>(); }

}  // namespace Test
