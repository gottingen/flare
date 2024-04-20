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

#include <Array.hpp>
#include <err_cuda.hpp>

#include <type_traits>

namespace flare {
namespace cuda {

template<typename T>
class LookupTable1D {
   public:
    LookupTable1D()                                     = delete;
    LookupTable1D(const LookupTable1D& arg)             = delete;
    LookupTable1D(const LookupTable1D&& arg)            = delete;
    LookupTable1D& operator=(const LookupTable1D& arg)  = delete;
    LookupTable1D& operator=(const LookupTable1D&& arg) = delete;

    LookupTable1D(const Array<T>& lutArray) : mTexture(0), mData(lutArray) {
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));

        resDesc.resType                = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr      = mData.get();
        resDesc.res.linear.desc.x      = sizeof(T) * 8;
        resDesc.res.linear.sizeInBytes = mData.elements() * sizeof(T);

        if (std::is_signed<T>::value)
            resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        else if (std::is_unsigned<T>::value)
            resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        else
            resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;

        texDesc.readMode = cudaReadModeElementType;

        CUDA_CHECK(
            cudaCreateTextureObject(&mTexture, &resDesc, &texDesc, NULL));
    }

    ~LookupTable1D() {
        if (mTexture) { cudaDestroyTextureObject(mTexture); }
    }

    cudaTextureObject_t get() const noexcept { return mTexture; }

   private:
    // Keep a copy so that ref count doesn't go down to zero when
    // original Array<T> goes out of scope before LookupTable1D object does.
    Array<T> mData;
    cudaTextureObject_t mTexture;
};

}  // namespace cuda
}  // namespace flare
