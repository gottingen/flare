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

#include <flare.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <sstream>
#include <vector>

using fly::array;
using fly::dim4;
using fly::getAvailableBackends;
using fly::randomEngine;
using fly::randu;
using fly::setBackend;
using fly::setSeed;
using std::get;
using std::make_pair;
using std::stringstream;
using std::vector;

enum param { engine, backend, size, seed, type };

using rng_params =
    std::tuple<fly::randomEngineType, std::pair<fly::Backend, fly::Backend>,
               fly::dim4, int, fly_dtype>;

class RNGMatch : public ::testing::TestWithParam<rng_params> {
   protected:
    void SetUp() {
        backends_available =
            getAvailableBackends() & get<backend>(GetParam()).first;
        backends_available =
            backends_available &&
            (getAvailableBackends() & get<backend>(GetParam()).second);

        if (backends_available) {
            setBackend(get<backend>(GetParam()).first);
            randomEngine(get<engine>(GetParam()));
            setSeed(get<seed>(GetParam()));
            array tmp  = randu(get<size>(GetParam()), get<type>(GetParam()));
            void* data = malloc(tmp.bytes());
            tmp.host(data);

            setBackend(get<backend>(GetParam()).second);
            values[0] = array(get<size>(GetParam()), get<type>(GetParam()));
            values[0].write(data, values[0].bytes());
            free(data);
            randomEngine(get<engine>(GetParam()));
            setSeed(get<seed>(GetParam()));
            values[1] = randu(get<size>(GetParam()), get<type>(GetParam()));
        }
    }

    array values[2];
    bool backends_available;
};

std::string engine_name(fly::randomEngineType engine) {
    switch (engine) {
        case FLY_RANDOM_ENGINE_PHILOX: return "PHILOX";
        case FLY_RANDOM_ENGINE_THREEFRY: return "THREEFRY";
        case FLY_RANDOM_ENGINE_MERSENNE: return "MERSENNE";
        default: return "UNKNOWN ENGINE";
    }
}

std::string backend_name(fly::Backend backend) {
    switch (backend) {
        case FLY_BACKEND_DEFAULT: return "DEFAULT";
        case FLY_BACKEND_CPU: return "CPU";
        case FLY_BACKEND_CUDA: return "CUDA";
        default: return "UNKNOWN BACKEND";
    }
}

std::string rngmatch_info(
    const ::testing::TestParamInfo<RNGMatch::ParamType> info) {
    stringstream ss;
    ss << "size_" << get<size>(info.param)[0] << "_"
       << backend_name(get<backend>(info.param).first) << "_"
       << backend_name(get<backend>(info.param).second) << "_"
       << get<size>(info.param)[1] << "_" << get<size>(info.param)[2] << "_"
       << get<size>(info.param)[3] << "_seed_" << get<seed>(info.param)
       << "_type_" << get<type>(info.param);
    return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
        PhiloxCPU_CUDA, RNGMatch,
        ::testing::Combine(
                ::testing::Values(FLY_RANDOM_ENGINE_PHILOX),
                ::testing::Values(make_pair(FLY_BACKEND_CPU, FLY_BACKEND_CUDA)),
                ::testing::Values(dim4(10), dim4(100), dim4(1000), dim4(10000),
                                  dim4(1E5), dim4(10, 10), dim4(10, 100),
                                  dim4(100, 100), dim4(1000, 100), dim4(10, 10, 10),
                                  dim4(10, 100, 10), dim4(100, 100, 10),
                                  dim4(1000, 100, 10), dim4(10, 10, 10, 10),
                                  dim4(10, 100, 10, 10), dim4(100, 100, 10, 10),
                                  dim4(1000, 100, 10, 10)),
                ::testing::Values(12), ::testing::Values(f32, f64, c32, c64, u8)),
        rngmatch_info);

TEST_P(RNGMatch, BackendEquals) {
    if (backends_available) {
        array actual   = values[0];
        array expected = values[1];
        ASSERT_ARRAYS_EQ(actual, expected);
    } else {
        printf("SKIPPED\n");
    }
}
