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

#ifndef __MAGMA_HELPER_H
#define __MAGMA_HELPER_H

template<typename T>
T magma_zero();
template<typename T>
T magma_one();
template<typename T>
T magma_neg_one();
template<typename T>
T magma_scalar(double val);
template<typename T>
double magma_real(T val);
template<typename T>
T magma_make(double r, double i);

template<typename T>
bool magma_is_real();

template<typename T>
magma_int_t magma_get_getrf_nb(int num);
template<typename T>
magma_int_t magma_get_potrf_nb(int num);
template<typename T>
magma_int_t magma_get_geqrf_nb(int num);
template<typename T>
magma_int_t magma_get_gebrd_nb(int /*num*/) {
    return 32;
}

#endif
