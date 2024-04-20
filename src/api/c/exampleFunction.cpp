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

#include <fly/dim4.hpp>  // Needed if you use dim4 class

#include <fly/util.h>  // Include header where function is delcared

#include <fly/defines.h>  // Include this header to access any enums,
                         // #defines or constants declared

#include <common/err_common.hpp>  // Header with error checking functions & macros

#include <backend.hpp>  // This header make sures appropriate backend
                        // related namespace is being used

#include <Array.hpp>  // Header in which backend specific Array class
                      // is defined

#include <handle.hpp>  // Header that helps you retrieve backend specific
                       // Arrays based on the fly_array
                       // (typedef in defines.h) handle.

#include <exampleFunction.hpp>  // This is the backend specific header
                                // where your new function declaration
                                // is written

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace detail;  // detail is an alias to appropriate backend
                         // defined in backend.hpp. You don't need to
                         // change this

template<typename T>
fly_array example(const fly_array& a, const fly_array& b,
                 const fly_someenum_t& param) {
    // getArray<T> function is defined in handle.hpp
    // and it returns backend specific Array, namely one of the following
    //      * cpu::Array<T>
    //      * flare::cuda::Array<T>
    // getHandle<T> function is defined in handle.hpp takes one of the
    // above backend specific detail::Array<T> and returns the
    // universal array handle fly_array
    return getHandle<T>(exampleFunction(getArray<T>(a), getArray<T>(b), param));
}

fly_err fly_example_function(fly_array* out, const fly_array a,
                           const fly_someenum_t param) {
    try {
        fly_array output = 0;
        const ArrayInfo& info =
            getInfo(a);  // ArrayInfo is the base class which
                         // each backend specific Array inherits
                         // This class stores the basic array meta-data
                         // such as type of data, dimensions,
                         // offsets and strides. This class is declared
                         // in src/backend/common/ArrayInfo.hpp
        fly::dim4 dims = info.dims();

        ARG_ASSERT(2, (dims.ndims() >= 0 && dims.ndims() <= 3));
        // defined in err_common.hpp
        // there are other useful Macros
        // for different purposes, feel free
        // to look at the header

        fly_dtype type = info.getType();

        switch (type) {  // Based on the data type, call backend specific
                         // implementation
            case f64: output = example<double>(a, a, param); break;
            case f32: output = example<float>(a, a, param); break;
            case s32: output = example<int>(a, a, param); break;
            case u32: output = example<uint>(a, a, param); break;
            case u8: output = example<uchar>(a, a, param); break;
            case b8: output = example<char>(a, a, param); break;
            case c32: output = example<cfloat>(a, a, param); break;
            case c64: output = example<cdouble>(a, a, param); break;
            default:
                TYPE_ERROR(1,
                           type);  // Another helpful macro from err_common.hpp
                                   // that helps throw type based error messages
        }

        std::swap(*out, output);  // if the function has returned successfully,
                                  // swap the temporary 'output' variable with
                                  // '*out'
    }
    CATCHALL;  // All throws/exceptions from any internal
               // implementations are caught by this CATCHALL
               // macro and handled appropriately.

    return FLY_SUCCESS;  // In case of successfull completion, return FLY_SUCCESS
                        // There are set of error codes defined in defines.h
                        // which you are used by CATCHALL to return approriate
                        // code
}
