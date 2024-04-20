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

#include <fly/random.h>

#include <backend.hpp>
#include <common/MersenneTwister.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <random_engine.hpp>
#include <types.hpp>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/defines.h>
#include <fly/device.h>
#include <fly/dim4.hpp>
#include <map>
#include <memory>

using fly::dim4;
using flare::common::half;
using flare::common::mask;
using flare::common::MaxBlocks;
using flare::common::MtStateLength;
using flare::common::pos;
using flare::common::recursion_tbl;
using flare::common::sh1;
using flare::common::sh2;
using flare::common::TableLength;
using flare::common::temper_tbl;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::createHostDataArray;
using detail::intl;
using detail::normalDistribution;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::uniformDistribution;
using detail::ushort;

Array<uint> emptyArray() { return createEmptyArray<uint>(dim4(0)); }

struct RandomEngine {
    // clang-format off
    fly_random_engine_type type{FLY_RANDOM_ENGINE_DEFAULT}; // NOLINT(misc-non-private-member-variables-in-classes)
    std::shared_ptr<uintl> seed;                          // NOLINT(misc-non-private-member-variables-in-classes)
    std::shared_ptr<uintl> counter;                       // NOLINT(misc-non-private-member-variables-in-classes)
    Array<uint> pos;                                      // NOLINT(misc-non-private-member-variables-in-classes)
    Array<uint> sh1;                                      // NOLINT(misc-non-private-member-variables-in-classes)
    Array<uint> sh2;                                      // NOLINT(misc-non-private-member-variables-in-classes)
    uint mask{0};                                         // NOLINT(misc-non-private-member-variables-in-classes)
    Array<uint> recursion_table;                          // NOLINT(misc-non-private-member-variables-in-classes)
    Array<uint> temper_table;                             // NOLINT(misc-non-private-member-variables-in-classes)
    Array<uint> state;                                    // NOLINT(misc-non-private-member-variables-in-classes)
    // clang-format on

    RandomEngine()
        : seed(new uintl())
        , counter(new uintl())
        , pos(emptyArray())
        , sh1(emptyArray())
        , sh2(emptyArray())
        , recursion_table(emptyArray())
        , temper_table(emptyArray())
        , state(emptyArray()) {}
};

fly_random_engine getRandomEngineHandle(const RandomEngine &engine) {
    auto *engineHandle = new RandomEngine;
    *engineHandle      = engine;
    return static_cast<fly_random_engine>(engineHandle);
}

RandomEngine *getRandomEngine(const fly_random_engine engineHandle) {
    if (engineHandle == 0) {
        FLY_ERROR("Uninitialized random engine", FLY_ERR_ARG);
    }
    return static_cast<RandomEngine *>(engineHandle);
}

namespace {
template<typename T>
inline fly_array uniformDistribution_(const dim4 &dims, RandomEngine *e) {
    if (e->type == FLY_RANDOM_ENGINE_MERSENNE_GP11213) {
        return getHandle(uniformDistribution<T>(dims, e->pos, e->sh1, e->sh2,
                                                e->mask, e->recursion_table,
                                                e->temper_table, e->state));
    } else {
        return getHandle(
            uniformDistribution<T>(dims, e->type, *(e->seed), *(e->counter)));
    }
}

template<typename T>
inline fly_array normalDistribution_(const dim4 &dims, RandomEngine *e) {
    if (e->type == FLY_RANDOM_ENGINE_MERSENNE_GP11213) {
        return getHandle(normalDistribution<T>(dims, e->pos, e->sh1, e->sh2,
                                               e->mask, e->recursion_table,
                                               e->temper_table, e->state));
    } else {
        return getHandle(
            normalDistribution<T>(dims, e->type, *(e->seed), *(e->counter)));
    }
}

void validateRandomType(const fly_random_engine_type type) {
    if ((type != FLY_RANDOM_ENGINE_PHILOX_4X32_10) &&
        (type != FLY_RANDOM_ENGINE_THREEFRY_2X32_16) &&
        (type != FLY_RANDOM_ENGINE_MERSENNE_GP11213) &&
        (type != FLY_RANDOM_ENGINE_PHILOX) &&
        (type != FLY_RANDOM_ENGINE_THREEFRY) &&
        (type != FLY_RANDOM_ENGINE_MERSENNE) &&
        (type != FLY_RANDOM_ENGINE_DEFAULT)) {
        FLY_ERROR("Invalid random type", FLY_ERR_ARG);
    }
}
}  // namespace

fly_err fly_get_default_random_engine(fly_random_engine *r) {
    try {
        FLY_CHECK(fly_init());

        // RandomEngine contains device buffers which are dependent on
        // context|stream/device. Since nor context or stream are available at
        // this level, we will only use the deviceId.
        thread_local std::map<int /*deviceId*/, RandomEngine *>
            cachedDefaultRandomEngines;
        const int dependent = fly::getDevice();
        auto it             = cachedDefaultRandomEngines.find(dependent);
        if (it == cachedDefaultRandomEngines.end()) {
            RandomEngine *defaultRandomEngine     = new RandomEngine;
            cachedDefaultRandomEngines[dependent] = defaultRandomEngine;
            *r = static_cast<fly_random_engine>(defaultRandomEngine);
        } else {
            *r = static_cast<fly_random_engine>(it->second);
        }
        return FLY_SUCCESS;
    }
    CATCHALL;
}

fly_err fly_create_random_engine(fly_random_engine *engineHandle,
                               fly_random_engine_type rtype, uintl seed) {
    try {
        FLY_CHECK(fly_init());
        validateRandomType(rtype);

        RandomEngine e;
        e.type     = rtype;
        *e.seed    = seed;
        *e.counter = 0;

        if (rtype == FLY_RANDOM_ENGINE_MERSENNE_GP11213) {
            e.pos  = createHostDataArray<uint>(dim4(MaxBlocks), pos);
            e.sh1  = createHostDataArray<uint>(dim4(MaxBlocks), sh1);
            e.sh2  = createHostDataArray<uint>(dim4(MaxBlocks), sh2);
            e.mask = mask;

            e.recursion_table =
                createHostDataArray<uint>(dim4(TableLength), recursion_tbl);
            e.temper_table =
                createHostDataArray<uint>(dim4(TableLength), temper_tbl);
            e.state = createEmptyArray<uint>(dim4(MtStateLength));

            initMersenneState(e.state, seed, e.recursion_table);
        }

        *engineHandle = getRandomEngineHandle(e);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_retain_random_engine(fly_random_engine *outHandle,
                               const fly_random_engine engineHandle) {
    try {
        FLY_CHECK(fly_init());
        *outHandle = getRandomEngineHandle(*(getRandomEngine(engineHandle)));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_random_engine_set_type(fly_random_engine *engine,
                                 const fly_random_engine_type rtype) {
    try {
        FLY_CHECK(fly_init());
        validateRandomType(rtype);
        RandomEngine *e = getRandomEngine(*engine);
        if (rtype != e->type) {
            if (rtype == FLY_RANDOM_ENGINE_MERSENNE_GP11213) {
                e->pos  = createHostDataArray<uint>(dim4(MaxBlocks), pos);
                e->sh1  = createHostDataArray<uint>(dim4(MaxBlocks), sh1);
                e->sh2  = createHostDataArray<uint>(dim4(MaxBlocks), sh2);
                e->mask = mask;

                e->recursion_table =
                    createHostDataArray<uint>(dim4(TableLength), recursion_tbl);
                e->temper_table =
                    createHostDataArray<uint>(dim4(TableLength), temper_tbl);
                e->state = createEmptyArray<uint>(dim4(MtStateLength));

                initMersenneState(e->state, *(e->seed), e->recursion_table);
            } else if (e->type == FLY_RANDOM_ENGINE_MERSENNE_GP11213) {
                e->pos             = emptyArray();
                e->sh1             = emptyArray();
                e->sh2             = emptyArray();
                e->mask            = 0;
                e->recursion_table = emptyArray();
                e->temper_table    = emptyArray();
                e->state           = emptyArray();
            }
            e->type = rtype;
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_random_engine_get_type(fly_random_engine_type *rtype,
                                 const fly_random_engine engine) {
    try {
        FLY_CHECK(fly_init());
        RandomEngine *e = getRandomEngine(engine);
        *rtype          = e->type;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_default_random_engine_type(const fly_random_engine_type rtype) {
    try {
        FLY_CHECK(fly_init());
        fly_random_engine e;
        FLY_CHECK(fly_get_default_random_engine(&e));
        FLY_CHECK(fly_random_engine_set_type(&e, rtype));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_random_engine_set_seed(fly_random_engine *engine, const uintl seed) {
    try {
        FLY_CHECK(fly_init());
        RandomEngine *e = getRandomEngine(*engine);
        *(e->seed)      = seed;
        if (e->type == FLY_RANDOM_ENGINE_MERSENNE_GP11213) {
            initMersenneState(e->state, seed, e->recursion_table);
        } else {
            *(e->counter) = 0;
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_random_engine_get_seed(uintl *const seed, fly_random_engine engine) {
    try {
        FLY_CHECK(fly_init());
        RandomEngine *e = getRandomEngine(engine);
        *seed           = *(e->seed);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_random_uniform(fly_array *out, const unsigned ndims,
                         const dim_t *const dims, const fly_dtype type,
                         fly_random_engine engine) {
    try {
        FLY_CHECK(fly_init());
        fly_array result;

        dim4 d          = verifyDims(ndims, dims);
        RandomEngine *e = getRandomEngine(engine);

        switch (type) {
            case f32: result = uniformDistribution_<float>(d, e); break;
            case c32: result = uniformDistribution_<cfloat>(d, e); break;
            case f64: result = uniformDistribution_<double>(d, e); break;
            case c64: result = uniformDistribution_<cdouble>(d, e); break;
            case s32: result = uniformDistribution_<int>(d, e); break;
            case u32: result = uniformDistribution_<uint>(d, e); break;
            case s64: result = uniformDistribution_<intl>(d, e); break;
            case u64: result = uniformDistribution_<uintl>(d, e); break;
            case s16: result = uniformDistribution_<short>(d, e); break;
            case u16: result = uniformDistribution_<ushort>(d, e); break;
            case u8: result = uniformDistribution_<uchar>(d, e); break;
            case b8: result = uniformDistribution_<char>(d, e); break;
            case f16: result = uniformDistribution_<half>(d, e); break;
            default: TYPE_ERROR(4, type);
        }
        std::swap(*out, result);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_random_normal(fly_array *out, const unsigned ndims,
                        const dim_t *const dims, const fly_dtype type,
                        fly_random_engine engine) {
    try {
        FLY_CHECK(fly_init());
        fly_array result;

        dim4 d          = verifyDims(ndims, dims);
        RandomEngine *e = getRandomEngine(engine);

        switch (type) {
            case f32: result = normalDistribution_<float>(d, e); break;
            case c32: result = normalDistribution_<cfloat>(d, e); break;
            case f64: result = normalDistribution_<double>(d, e); break;
            case c64: result = normalDistribution_<cdouble>(d, e); break;
            case f16: result = normalDistribution_<half>(d, e); break;
            default: TYPE_ERROR(4, type);
        }
        std::swap(*out, result);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_release_random_engine(fly_random_engine engineHandle) {
    try {
        FLY_CHECK(fly_init());
        delete getRandomEngine(engineHandle);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_randu(fly_array *out, const unsigned ndims, const dim_t *const dims,
                const fly_dtype type) {
    try {
        FLY_CHECK(fly_init());
        fly_array result;

        fly_random_engine engine;
        FLY_CHECK(fly_get_default_random_engine(&engine));
        RandomEngine *e = getRandomEngine(engine);
        dim4 d          = verifyDims(ndims, dims);

        switch (type) {
            case f32: result = uniformDistribution_<float>(d, e); break;
            case c32: result = uniformDistribution_<cfloat>(d, e); break;
            case f64: result = uniformDistribution_<double>(d, e); break;
            case c64: result = uniformDistribution_<cdouble>(d, e); break;
            case s32: result = uniformDistribution_<int>(d, e); break;
            case u32: result = uniformDistribution_<uint>(d, e); break;
            case s64: result = uniformDistribution_<intl>(d, e); break;
            case u64: result = uniformDistribution_<uintl>(d, e); break;
            case s16: result = uniformDistribution_<short>(d, e); break;
            case u16: result = uniformDistribution_<ushort>(d, e); break;
            case u8: result = uniformDistribution_<uchar>(d, e); break;
            case b8: result = uniformDistribution_<char>(d, e); break;
            case f16: result = uniformDistribution_<half>(d, e); break;
            default: TYPE_ERROR(3, type);
        }
        std::swap(*out, result);
    }
    CATCHALL
    return FLY_SUCCESS;
}

fly_err fly_randn(fly_array *out, const unsigned ndims, const dim_t *const dims,
                const fly_dtype type) {
    try {
        FLY_CHECK(fly_init());
        fly_array result;

        fly_random_engine engine;
        FLY_CHECK(fly_get_default_random_engine(&engine));
        RandomEngine *e = getRandomEngine(engine);
        dim4 d          = verifyDims(ndims, dims);

        switch (type) {
            case f32: result = normalDistribution_<float>(d, e); break;
            case c32: result = normalDistribution_<cfloat>(d, e); break;
            case f64: result = normalDistribution_<double>(d, e); break;
            case c64: result = normalDistribution_<cdouble>(d, e); break;
            case f16: result = normalDistribution_<half>(d, e); break;
            default: TYPE_ERROR(3, type);
        }
        std::swap(*out, result);
    }
    CATCHALL
    return FLY_SUCCESS;
}

fly_err fly_set_seed(const uintl seed) {
    try {
        FLY_CHECK(fly_init());
        fly_random_engine engine;
        FLY_CHECK(fly_get_default_random_engine(&engine));
        FLY_CHECK(fly_random_engine_set_seed(&engine, seed));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_seed(uintl *seed) {
    try {
        FLY_CHECK(fly_init());
        fly_random_engine e;
        FLY_CHECK(fly_get_default_random_engine(&e));
        FLY_CHECK(fly_random_engine_get_seed(seed, e));
    }
    CATCHALL;
    return FLY_SUCCESS;
}
