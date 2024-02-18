/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/data.h>
#include <fly/dim4.hpp>
#include <fly/random.h>
#include "error.hpp"

namespace fly {
randomEngine::randomEngine(randomEngineType type, unsigned long long seed)
    : engine(0) {
    FLY_THROW(fly_create_random_engine(&engine, type, seed));
}

randomEngine::randomEngine(const randomEngine &other) : engine(0) {
    if (this != &other) {
        FLY_THROW(fly_retain_random_engine(&engine, other.get()));
    }
}

randomEngine::randomEngine(fly_random_engine engine) : engine(engine) {}

randomEngine::~randomEngine() {
    if (engine) { fly_release_random_engine(engine); }
}

randomEngine &randomEngine::operator=(const randomEngine &other) {
    if (this != &other) {
        FLY_THROW(fly_release_random_engine(engine));
        FLY_THROW(fly_retain_random_engine(&engine, other.get()));
    }
    return *this;
}

randomEngineType randomEngine::getType() {
    fly_random_engine_type type;
    FLY_THROW(fly_random_engine_get_type(&type, engine));
    return type;
}

void randomEngine::setType(const randomEngineType type) {
    FLY_THROW(fly_random_engine_set_type(&engine, type));
}

void randomEngine::setSeed(const unsigned long long seed) {
    FLY_THROW(fly_random_engine_set_seed(&engine, seed));
}

unsigned long long randomEngine::getSeed() const {
    unsigned long long seed;
    FLY_THROW(fly_random_engine_get_seed(&seed, engine));
    return seed;
}

fly_random_engine randomEngine::get() const { return engine; }

array randu(const dim4 &dims, const dtype ty, randomEngine &r) {
    fly_array out;
    FLY_THROW(fly_random_uniform(&out, dims.ndims(), dims.get(), ty, r.get()));
    return array(out);
}

array randn(const dim4 &dims, const dtype ty, randomEngine &r) {
    fly_array out;
    FLY_THROW(fly_random_normal(&out, dims.ndims(), dims.get(), ty, r.get()));
    return array(out);
}

array randu(const dim4 &dims, const fly::dtype type) {
    fly_array res;
    FLY_THROW(fly_randu(&res, dims.ndims(), dims.get(), type));
    return array(res);
}

array randu(const dim_t d0, const fly::dtype ty) { return randu(dim4(d0), ty); }

array randu(const dim_t d0, const dim_t d1, const fly::dtype ty) {
    return randu(dim4(d0, d1), ty);
}

array randu(const dim_t d0, const dim_t d1, const dim_t d2,
            const fly::dtype ty) {
    return randu(dim4(d0, d1, d2), ty);
}

array randu(const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3,
            const fly::dtype ty) {
    return randu(dim4(d0, d1, d2, d3), ty);
}

array randn(const dim4 &dims, const fly::dtype type) {
    fly_array res;
    FLY_THROW(fly_randn(&res, dims.ndims(), dims.get(), type));
    return array(res);
}

array randn(const dim_t d0, const fly::dtype ty) { return randn(dim4(d0), ty); }

array randn(const dim_t d0, const dim_t d1, const fly::dtype ty) {
    return randn(dim4(d0, d1), ty);
}

array randn(const dim_t d0, const dim_t d1, const dim_t d2,
            const fly::dtype ty) {
    return randn(dim4(d0, d1, d2), ty);
}

array randn(const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3,
            const fly::dtype ty) {
    return randn(dim4(d0, d1, d2, d3), ty);
}

void setDefaultRandomEngineType(randomEngineType rtype) {
    FLY_THROW(fly_set_default_random_engine_type(rtype));
}

randomEngine getDefaultRandomEngine() {
    fly_random_engine internal_handle = 0;
    fly_random_engine handle          = 0;
    FLY_THROW(fly_get_default_random_engine(&internal_handle));
    FLY_THROW(fly_retain_random_engine(&handle, internal_handle));
    return randomEngine(handle);
}

void setSeed(const unsigned long long seed) { FLY_THROW(fly_set_seed(seed)); }

unsigned long long getSeed() {
    unsigned long long seed = 0;
    FLY_THROW(fly_get_seed(&seed));
    return seed;
}

}  // namespace fly
