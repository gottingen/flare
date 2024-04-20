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

#include <Param.hpp>

namespace flare {
namespace cuda {

template<typename inType, typename outType, fly::matchType mType, bool needMean>
__global__ void matchTemplate(Param<outType> out, CParam<inType> srch,
                              CParam<inType> tmplt, int nBBS0, int nBBS1) {
    unsigned b2 = blockIdx.x / nBBS0;
    unsigned b3 = blockIdx.y / nBBS1;

    int gx = threadIdx.x + (blockIdx.x - b2 * nBBS0) * blockDim.x;
    int gy = threadIdx.y + (blockIdx.y - b3 * nBBS1) * blockDim.y;

    if (gx < srch.dims[0] && gy < srch.dims[1]) {
        const int tDim0    = tmplt.dims[0];
        const int tDim1    = tmplt.dims[1];
        const int sDim0    = srch.dims[0];
        const int sDim1    = srch.dims[1];
        const inType* tptr = (const inType*)tmplt.ptr;
        int winNumElems    = tDim0 * tDim1;

        outType tImgMean = outType(0);
        if (needMean) {
            for (int tj = 0; tj < tDim1; tj++) {
                int tjStride = tj * tmplt.strides[1];

                for (int ti = 0; ti < tDim0; ti++) {
                    tImgMean += (outType)tptr[tjStride + ti * tmplt.strides[0]];
                }
            }
            tImgMean /= winNumElems;
        }

        const inType* sptr = (const inType*)srch.ptr +
                             (b2 * srch.strides[2] + b3 * srch.strides[3]);
        outType* optr =
            (outType*)out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);

        // mean for window
        // this variable will be used based on mType value
        outType wImgMean = outType(0);
        if (needMean) {
            for (int tj = 0, j = gy; tj < tDim1; tj++, j++) {
                int jStride = j * srch.strides[1];

                for (int ti = 0, i = gx; ti < tDim0; ti++, i++) {
                    inType sVal = ((j < sDim1 && i < sDim0)
                                       ? sptr[jStride + i * srch.strides[0]]
                                       : inType(0));
                    wImgMean += (outType)sVal;
                }
            }
            wImgMean /= winNumElems;
        }

        // run the window match metric
        outType disparity = outType(0);

        for (int tj = 0, j = gy; tj < tDim1; tj++, j++) {
            int jStride  = j * srch.strides[1];
            int tjStride = tj * tmplt.strides[1];

            for (int ti = 0, i = gx; ti < tDim0; ti++, i++) {
                inType sVal = ((j < sDim1 && i < sDim0)
                                   ? sptr[jStride + i * srch.strides[0]]
                                   : inType(0));
                inType tVal = tptr[tjStride + ti * tmplt.strides[0]];

                outType temp;
                switch (mType) {
                    case FLY_SAD:
                        disparity += fabs((outType)sVal - (outType)tVal);
                        break;
                    case FLY_ZSAD:
                        disparity += fabs((outType)sVal - wImgMean -
                                          (outType)tVal + tImgMean);
                        break;
                    case FLY_LSAD:
                        disparity +=
                            fabs((outType)sVal - (wImgMean / tImgMean) * tVal);
                        break;
                    case FLY_SSD:
                        disparity += ((outType)sVal - (outType)tVal) *
                                     ((outType)sVal - (outType)tVal);
                        break;
                    case FLY_ZSSD:
                        temp = ((outType)sVal - wImgMean - (outType)tVal +
                                tImgMean);
                        disparity += temp * temp;
                        break;
                    case FLY_LSSD:
                        temp = ((outType)sVal - (wImgMean / tImgMean) * tVal);
                        disparity += temp * temp;
                        break;
                    case FLY_NCC:
                        // TODO: furture implementation
                        break;
                    case FLY_ZNCC:
                        // TODO: furture implementation
                        break;
                    case FLY_SHD:
                        // TODO: furture implementation
                        break;
                }
            }
        }
        optr[gy * out.strides[1] + gx] = disparity;
    }
}

}  // namespace cuda
}  // namespace flare
