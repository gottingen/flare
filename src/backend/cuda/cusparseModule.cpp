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

#include <cusparse.hpp>
#include <cusparseModule.hpp>

#include <common/Version.hpp>
#include <common/err_common.hpp>
#include <cusparse.hpp>
#include <platform.hpp>
#include <fly/defines.h>

#include <cuda.h>
#include <string>

using flare::common::Version;

namespace flare {
namespace cuda {

common::Version getCusparseVersion(const LibHandle& handle) {
    std::function<cusparseStatus_t(libraryPropertyType, int*)> fptr(
        reinterpret_cast<cusparseStatus_t (*)(libraryPropertyType, int*)>(
            common::getFunctionPointer(handle, "cusparseGetProperty")));

    int major, minor, patch;
    CUSPARSE_CHECK(fptr(MAJOR_VERSION, &major));
    CUSPARSE_CHECK(fptr(MINOR_VERSION, &minor));
    CUSPARSE_CHECK(fptr(PATCH_LEVEL, &patch));

    Version out{major, minor, patch};
    return out;
}

cusparseModule::cusparseModule()
    :
#ifdef FLY_cusparse_STATIC_LINKING
    module(nullptr, nullptr)
#else
    module({"cusparse"}, {"64_12", "64_11", "64_10", "64_9", "64_8"}, {""}, 0,
           nullptr, getCusparseVersion)
#endif
{
#ifdef FLY_cusparse_STATIC_LINKING
    FLY_TRACE("CuSparse linked staticly.");
#undef MODULE_FUNCTION_INIT
#define MODULE_FUNCTION_INIT(NAME) NAME = &::NAME
#else
    if (!module.isLoaded()) {
        FLY_TRACE(
            "WARNING: Unable to load cuSparse: {}\n"
            "cuSparse failed to load. Try installing cuSparse or check if\n"
            "cuSparse is in the search path. On Linux, you can set the\n"
            "LD_DEBUG=libs environment variable to debug loading issues.\n"
            "Falling back to matmul based implementation",
            module.getErrorMessage());

        return;
    }
#endif

    MODULE_FUNCTION_INIT(cusparseGetVersion);

#if CUSPARSE_VERSION < 11300
    MODULE_FUNCTION_INIT(cusparseCcsc2dense);
    MODULE_FUNCTION_INIT(cusparseCcsr2dense);
    MODULE_FUNCTION_INIT(cusparseCdense2csc);
    MODULE_FUNCTION_INIT(cusparseCdense2csr);
    MODULE_FUNCTION_INIT(cusparseCgthr);
    MODULE_FUNCTION_INIT(cusparseDcsc2dense);
    MODULE_FUNCTION_INIT(cusparseDcsr2dense);
    MODULE_FUNCTION_INIT(cusparseDdense2csc);
    MODULE_FUNCTION_INIT(cusparseDdense2csr);
    MODULE_FUNCTION_INIT(cusparseDgthr);
    MODULE_FUNCTION_INIT(cusparseScsc2dense);
    MODULE_FUNCTION_INIT(cusparseScsr2dense);
    MODULE_FUNCTION_INIT(cusparseSdense2csc);
    MODULE_FUNCTION_INIT(cusparseSdense2csr);
    MODULE_FUNCTION_INIT(cusparseSgthr);
    MODULE_FUNCTION_INIT(cusparseZcsc2dense);
    MODULE_FUNCTION_INIT(cusparseZcsr2dense);
    MODULE_FUNCTION_INIT(cusparseZdense2csc);
    MODULE_FUNCTION_INIT(cusparseZdense2csr);
    MODULE_FUNCTION_INIT(cusparseZgthr);
#else
    MODULE_FUNCTION_INIT(cusparseCreateCsc);
    MODULE_FUNCTION_INIT(cusparseSparseToDense_bufferSize);
    MODULE_FUNCTION_INIT(cusparseSparseToDense);
    MODULE_FUNCTION_INIT(cusparseDenseToSparse_bufferSize);
    MODULE_FUNCTION_INIT(cusparseDenseToSparse_analysis);
    MODULE_FUNCTION_INIT(cusparseDenseToSparse_convert);
    MODULE_FUNCTION_INIT(cusparseSpMatGetSize);
    MODULE_FUNCTION_INIT(cusparseCsrSetPointers);
    MODULE_FUNCTION_INIT(cusparseCscSetPointers);
    MODULE_FUNCTION_INIT(cusparseSetPointerMode);
    MODULE_FUNCTION_INIT(cusparseXcsrsort_bufferSizeExt);
    MODULE_FUNCTION_INIT(cusparseXcsrsort);
#endif

    MODULE_FUNCTION_INIT(cusparseCnnz);
    MODULE_FUNCTION_INIT(cusparseCreateCsr);
    MODULE_FUNCTION_INIT(cusparseCreateCoo);
    MODULE_FUNCTION_INIT(cusparseCreateDnMat);
    MODULE_FUNCTION_INIT(cusparseCreateDnVec);
    MODULE_FUNCTION_INIT(cusparseCreateIdentityPermutation);
    MODULE_FUNCTION_INIT(cusparseCreate);
    MODULE_FUNCTION_INIT(cusparseCreateMatDescr);
    MODULE_FUNCTION_INIT(cusparseDestroyDnMat);
    MODULE_FUNCTION_INIT(cusparseDestroyDnVec);
    MODULE_FUNCTION_INIT(cusparseDestroy);
    MODULE_FUNCTION_INIT(cusparseDestroyMatDescr);
    MODULE_FUNCTION_INIT(cusparseDestroySpMat);
    MODULE_FUNCTION_INIT(cusparseDnnz);
    MODULE_FUNCTION_INIT(cusparseSetMatIndexBase);
    MODULE_FUNCTION_INIT(cusparseSetMatType);
    MODULE_FUNCTION_INIT(cusparseSetStream);
    MODULE_FUNCTION_INIT(cusparseSnnz);
    MODULE_FUNCTION_INIT(cusparseSpMM_bufferSize);
    MODULE_FUNCTION_INIT(cusparseSpMM);
    MODULE_FUNCTION_INIT(cusparseSpMV_bufferSize);
    MODULE_FUNCTION_INIT(cusparseSpMV);
    MODULE_FUNCTION_INIT(cusparseXcoo2csr);
    MODULE_FUNCTION_INIT(cusparseXcoosort_bufferSizeExt);
    MODULE_FUNCTION_INIT(cusparseXcoosortByColumn);
    MODULE_FUNCTION_INIT(cusparseXcoosortByRow);
    MODULE_FUNCTION_INIT(cusparseXcsr2coo);
#if CUSPARSE_VERSION < 11000
    MODULE_FUNCTION_INIT(cusparseXcsrgeamNnz);
    MODULE_FUNCTION_INIT(cusparseScsrgeam);
    MODULE_FUNCTION_INIT(cusparseDcsrgeam);
    MODULE_FUNCTION_INIT(cusparseCcsrgeam);
    MODULE_FUNCTION_INIT(cusparseZcsrgeam);
#else
    MODULE_FUNCTION_INIT(cusparseXcsrgeam2Nnz);
    MODULE_FUNCTION_INIT(cusparseScsrgeam2_bufferSizeExt);
    MODULE_FUNCTION_INIT(cusparseScsrgeam2);
    MODULE_FUNCTION_INIT(cusparseDcsrgeam2_bufferSizeExt);
    MODULE_FUNCTION_INIT(cusparseDcsrgeam2);
    MODULE_FUNCTION_INIT(cusparseCcsrgeam2_bufferSizeExt);
    MODULE_FUNCTION_INIT(cusparseCcsrgeam2);
    MODULE_FUNCTION_INIT(cusparseZcsrgeam2_bufferSizeExt);
    MODULE_FUNCTION_INIT(cusparseZcsrgeam2);
#endif
    MODULE_FUNCTION_INIT(cusparseZnnz);

#ifndef FLY_cusparse_STATIC_LINKING
    if (!module.symbolsLoaded()) {
        std::string error_message =
            "Error loading cuSparse symbols. Flare was unable to load some "
            "symbols from the cuSparse library. Please create an issue on the "
            "Flare repository with information about the installed "
            "cuSparse and Flare on your system.";
        FLY_ERROR(error_message, FLY_ERR_LOAD_LIB);
    }
#endif
}

clog::logger* cusparseModule::getLogger() const noexcept {
    return module.getLogger();
}

cusparseModule& getCusparsePlugin() noexcept {
    static auto* plugin = new cusparseModule();
    return *plugin;
}

}  // namespace cuda
}  // namespace flare
