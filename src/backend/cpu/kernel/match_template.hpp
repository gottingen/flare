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

namespace flare {
namespace cpu {
namespace kernel {

template<typename OutT, typename InT, fly::matchType MatchType>
void matchTemplate(Param<OutT> out, CParam<InT> sImg, CParam<InT> tImg) {
    constexpr bool needMean = MatchType == FLY_ZSAD || MatchType == FLY_LSAD ||
                              MatchType == FLY_ZSSD || MatchType == FLY_LSSD ||
                              MatchType == FLY_ZNCC;

    const fly::dim4 sDims    = sImg.dims();
    const fly::dim4 tDims    = tImg.dims();
    const fly::dim4 sStrides = sImg.strides();
    const fly::dim4 tStrides = tImg.strides();

    const dim_t tDim0 = tDims[0];
    const dim_t tDim1 = tDims[1];
    const dim_t sDim0 = sDims[0];
    const dim_t sDim1 = sDims[1];

    const fly::dim4 oStrides = out.strides();

    OutT tImgMean        = OutT(0);
    dim_t winNumElements = tImg.dims().elements();
    const InT* tpl       = tImg.get();

    if (needMean) {
        for (dim_t tj = 0; tj < tDim1; tj++) {
            dim_t tjStride = tj * tStrides[1];

            for (dim_t ti = 0; ti < tDim0; ti++) {
                tImgMean += (OutT)tpl[tjStride + ti * tStrides[0]];
            }
        }
        tImgMean /= winNumElements;
    }

    OutT* dst      = out.get();
    const InT* src = sImg.get();

    for (dim_t b3 = 0; b3 < sDims[3]; ++b3) {
        for (dim_t b2 = 0; b2 < sDims[2]; ++b2) {
            // slide through image window after window
            for (dim_t sj = 0; sj < sDim1; sj++) {
                dim_t ojStride = sj * oStrides[1];

                for (dim_t si = 0; si < sDim0; si++) {
                    OutT disparity = OutT(0);

                    // mean for window
                    // this variable will be used based on MatchType value
                    OutT wImgMean = OutT(0);
                    if (needMean) {
                        for (dim_t tj = 0, j = sj; tj < tDim1; tj++, j++) {
                            dim_t jStride = j * sStrides[1];

                            for (dim_t ti = 0, i = si; ti < tDim0; ti++, i++) {
                                InT sVal = ((j < sDim1 && i < sDim0)
                                                ? src[jStride + i * sStrides[0]]
                                                : InT(0));
                                wImgMean += (OutT)sVal;
                            }
                        }
                        wImgMean /= winNumElements;
                    }

                    // run the window match metric
                    for (dim_t tj = 0, j = sj; tj < tDim1; tj++, j++) {
                        dim_t jStride  = j * sStrides[1];
                        dim_t tjStride = tj * tStrides[1];

                        for (dim_t ti = 0, i = si; ti < tDim0; ti++, i++) {
                            InT sVal = ((j < sDim1 && i < sDim0)
                                            ? src[jStride + i * sStrides[0]]
                                            : InT(0));
                            InT tVal = tpl[tjStride + ti * tStrides[0]];
                            OutT temp;
                            switch (MatchType) {
                                case FLY_SAD:
                                    disparity += fabs((OutT)sVal - (OutT)tVal);
                                    break;
                                case FLY_ZSAD:
                                    disparity += fabs((OutT)sVal - wImgMean -
                                                      (OutT)tVal + tImgMean);
                                    break;
                                case FLY_LSAD:
                                    disparity +=
                                        fabs((OutT)sVal -
                                             (wImgMean / tImgMean) * tVal);
                                    break;
                                case FLY_SSD:
                                    disparity += ((OutT)sVal - (OutT)tVal) *
                                                 ((OutT)sVal - (OutT)tVal);
                                    break;
                                case FLY_ZSSD:
                                    temp = ((OutT)sVal - wImgMean - (OutT)tVal +
                                            tImgMean);
                                    disparity += temp * temp;
                                    break;
                                case FLY_LSSD:
                                    temp = ((OutT)sVal -
                                            (wImgMean / tImgMean) * tVal);
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
                    // output is just created, hence not doing the
                    // extra multiplication for 0th dim stride
                    dst[ojStride + si] = disparity;
                }
            }
            src += sStrides[2];
            dst += oStrides[2];
        }
        src += sStrides[3];
        dst += oStrides[3];
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
