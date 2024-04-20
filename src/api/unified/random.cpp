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
#include <fly/random.h>
#include "symbol_manager.hpp"

fly_err fly_get_default_random_engine(fly_random_engine *r) {
    CALL(fly_get_default_random_engine, r);
}

fly_err fly_create_random_engine(fly_random_engine *engineHandle,
                               fly_random_engine_type rtype,
                               unsigned long long seed) {
    CALL(fly_create_random_engine, engineHandle, rtype, seed);
}

fly_err fly_retain_random_engine(fly_random_engine *outHandle,
                               const fly_random_engine engineHandle) {
    CALL(fly_retain_random_engine, outHandle, engineHandle);
}

fly_err fly_random_engine_get_type(fly_random_engine_type *rtype,
                                 const fly_random_engine engine) {
    CALL(fly_random_engine_get_type, rtype, engine);
}

fly_err fly_random_engine_set_type(fly_random_engine *engine,
                                 const fly_random_engine_type rtype) {
    CALL(fly_random_engine_set_type, engine, rtype);
}

fly_err fly_set_default_random_engine_type(const fly_random_engine_type rtype) {
    CALL(fly_set_default_random_engine_type, rtype);
}

fly_err fly_random_uniform(fly_array *arr, const unsigned ndims,
                         const dim_t *const dims, const fly_dtype type,
                         fly_random_engine engine) {
    CALL(fly_random_uniform, arr, ndims, dims, type, engine);
}

fly_err fly_random_normal(fly_array *arr, const unsigned ndims,
                        const dim_t *const dims, const fly_dtype type,
                        fly_random_engine engine) {
    CALL(fly_random_normal, arr, ndims, dims, type, engine);
}

fly_err fly_release_random_engine(fly_random_engine engineHandle) {
    CALL(fly_release_random_engine, engineHandle);
}

fly_err fly_random_engine_set_seed(fly_random_engine *engine,
                                 const unsigned long long seed) {
    CALL(fly_random_engine_set_seed, engine, seed);
}

fly_err fly_random_engine_get_seed(unsigned long long *const seed,
                                 fly_random_engine engine) {
    CALL(fly_random_engine_get_seed, seed, engine);
}

fly_err fly_randu(fly_array *out, const unsigned ndims, const dim_t *const dims,
                const fly_dtype type) {
    CALL(fly_randu, out, ndims, dims, type);
}

fly_err fly_randn(fly_array *out, const unsigned ndims, const dim_t *const dims,
                const fly_dtype type) {
    CALL(fly_randn, out, ndims, dims, type);
}

fly_err fly_set_seed(const unsigned long long seed) { CALL(fly_set_seed, seed); }

fly_err fly_get_seed(unsigned long long *seed) { CALL(fly_get_seed, seed); }
