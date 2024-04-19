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

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <fly/defines.h>

#include <string>
#include <vector>

namespace flare {
namespace oneapi {
namespace kernel {

constexpr int THREADS = 256;

template<typename T, typename convT>
void calcParamSizes(Param<T>& sig_tmp, Param<T>& filter_tmp,
                    Param<convT>& packed, Param<T>& sig, Param<T>& filter,
                    const int rank, FLY_BATCH_KIND kind) {
    sig_tmp.info.dims[0] = filter_tmp.info.dims[0] = packed.info.dims[0];
    sig_tmp.info.strides[0] = filter_tmp.info.strides[0] = 1;

    for (int k = 1; k < 4; k++) {
        if (k < rank) {
            sig_tmp.info.dims[k]    = packed.info.dims[k];
            filter_tmp.info.dims[k] = packed.info.dims[k];
        } else {
            sig_tmp.info.dims[k]    = sig.info.dims[k];
            filter_tmp.info.dims[k] = filter.info.dims[k];
        }

        sig_tmp.info.strides[k] =
            sig_tmp.info.strides[k - 1] * sig_tmp.info.dims[k - 1];
        filter_tmp.info.strides[k] =
            filter_tmp.info.strides[k - 1] * filter_tmp.info.dims[k - 1];
    }

    // NOTE: The OpenCL implementation on which this oneAPI port is
    // based treated the incoming `packed` buffer as a string of real
    // scalars instead of complex numbers. OpenCL accomplished this
    // with the hack depicted in the trailing two lines. This note
    // remains here in an explanation of SYCL buffer reinterpret's in
    // fftconvolve kernel invocations.

    // sig_tmp.data    = packed.data;
    // filter_tmp.data = packed.data;

    // Calculate memory offsets for packed signal and filter
    if (kind == FLY_BATCH_RHS) {
        filter_tmp.info.offset = 0;
        sig_tmp.info.offset =
            filter_tmp.info.strides[3] * filter_tmp.info.dims[3] * 2;
    } else {
        sig_tmp.info.offset = 0;
        filter_tmp.info.offset =
            sig_tmp.info.strides[3] * sig_tmp.info.dims[3] * 2;
    }
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace flare
