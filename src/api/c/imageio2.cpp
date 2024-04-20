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

#if defined(WITH_FREEIMAGE)

#include "imageio_helper.h"

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <memory.hpp>
#include <traits.hpp>
#include <fly/algorithm.h>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/blas.h>
#include <fly/data.h>
#include <fly/dim4.hpp>
#include <fly/image.h>
#include <fly/index.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using fly::dim4;
using flare::FLYFI_GRAY;
using flare::FLYFI_RGB;
using flare::FLYFI_RGBA;
using flare::bitmap_ptr;
using flare::channel_split;
using flare::FI_CHANNELS;
using flare::FreeImage_Module;
using flare::FreeImageErrorHandler;
using flare::getFreeImagePlugin;
using flare::make_bitmap_ptr;
using detail::pinnedAlloc;
using detail::pinnedFree;
using detail::uchar;
using detail::uint;
using detail::ushort;

namespace {
template<typename T, FI_CHANNELS fi_color>
static fly_err readImage_t(fly_array* rImage, const uchar* pSrcLine,
                          const int nSrcPitch, const uint fi_w,
                          const uint fi_h) {
    // create an array to receive the loaded image data.
    FLY_CHECK(fly_init());
    T* pDst  = pinnedAlloc<T>(fi_w * fi_h * 4);  // 4 channels is max
    T* pDst0 = pDst;
    T* pDst1 = pDst + (fi_w * fi_h * 1);
    T* pDst2 = pDst + (fi_w * fi_h * 2);
    T* pDst3 = pDst + (fi_w * fi_h * 3);

    uint indx = 0;
    uint step = fi_color;

    for (uint x = 0; x < fi_w; ++x) {
        for (uint y = 0; y < fi_h; ++y) {
            const T* src = reinterpret_cast<T*>(const_cast<uchar*>(pSrcLine) -
                                                y * nSrcPitch);
            if (fi_color == 1) {
                pDst0[indx] = *(src + (x * step));
            } else if (fi_color >= 3) {
                if (static_cast<fly_dtype>(fly::dtype_traits<T>::fly_type) == u8) {
                    pDst0[indx] = *(src + (x * step + FI_RGBA_RED));
                    pDst1[indx] = *(src + (x * step + FI_RGBA_GREEN));
                    pDst2[indx] = *(src + (x * step + FI_RGBA_BLUE));
                    if (fi_color == 4) {
                        pDst3[indx] = *(src + (x * step + FI_RGBA_ALPHA));
                    }
                } else {
                    // Non 8-bit types do not use ordering
                    // See Pixel Access Functions Chapter in FreeImage Doc
                    pDst0[indx] = *(src + (x * step + 0));
                    pDst1[indx] = *(src + (x * step + 1));
                    pDst2[indx] = *(src + (x * step + 2));
                    if (fi_color == 4) {
                        pDst3[indx] = *(src + (x * step + 3));
                    }
                }
            }
            indx++;
        }
    }

    fly::dim4 dims(fi_h, fi_w, fi_color, 1);
    fly_err err =
        fly_create_array(rImage, pDst, dims.ndims(), dims.get(),
                        static_cast<fly_dtype>(fly::dtype_traits<T>::fly_type));
    pinnedFree(pDst);
    return err;
}

FREE_IMAGE_TYPE getFIT(FI_CHANNELS channels, fly_dtype type) {
    if (channels == FLYFI_GRAY) {
        if (type == u8) { return FIT_BITMAP; }
        if (type == u16) {
            return FIT_UINT16;
        } else if (type == f32) {
            return FIT_FLOAT;
        }
    } else if (channels == FLYFI_RGB) {
        if (type == u8) { return FIT_BITMAP; }
        if (type == u16) {
            return FIT_RGB16;
        } else if (type == f32) {
            return FIT_RGBF;
        }
    } else if (channels == FLYFI_RGBA) {
        if (type == u8) { return FIT_BITMAP; }
        if (type == u16) {
            return FIT_RGBA16;
        } else if (type == f32) {
            return FIT_RGBAF;
        }
    }
    return FIT_BITMAP;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// File IO
////////////////////////////////////////////////////////////////////////////////
// Load image from disk.
fly_err fly_load_image_native(fly_array* out, const char* filename) {
    try {
        ARG_ASSERT(1, filename != NULL);

        FreeImage_Module& _ = getFreeImagePlugin();

        // set your own FreeImage error handler
        _.FreeImage_SetOutputMessage(FreeImageErrorHandler);

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = _.FreeImage_GetFileType(filename, 0);
        if (fif == FIF_UNKNOWN) {
            fif = _.FreeImage_GetFIFFromFilename(filename);
        }

        if (fif == FIF_UNKNOWN) {
            FLY_ERROR("FreeImage Error: Unknown File or Filetype",
                     FLY_ERR_NOT_SUPPORTED);
        }

        unsigned flags = 0;
        if (fif == FIF_JPEG) {
            flags = flags | static_cast<unsigned>(JPEG_ACCURATE);
        }

        // check that the plugin has reading capabilities ...
        bitmap_ptr pBitmap = make_bitmap_ptr(nullptr);
        if (_.FreeImage_FIFSupportsReading(fif)) {
            pBitmap.reset(
                _.FreeImage_Load(fif, filename, static_cast<int>(flags)));
        }

        if (pBitmap == NULL) {
            FLY_ERROR(
                "FreeImage Error: Error reading image or file does not exist",
                FLY_ERR_RUNTIME);
        }

        // check image color type
        uint color_type   = _.FreeImage_GetColorType(pBitmap.get());
        const uint fi_bpp = _.FreeImage_GetBPP(pBitmap.get());
        // int fi_color = (int)((fi_bpp / 8.0) + 0.5);        //ceil
        uint fi_color;
        switch (color_type) {
            case 0:  // FIC_MINISBLACK
            case 1:  // FIC_MINISWHITE
                fi_color = 1;
                break;
            case 2:  // FIC_PALETTE
            case 3:  // FIC_RGB
                fi_color = 3;
                break;
            case 4:  // FIC_RGBALPHA
            case 5:  // FIC_CMYK
                fi_color = 4;
                break;
            default:  // Should not come here
                fi_color = 3;
                break;
        }

        const uint fi_bpc = fi_bpp / fi_color;
        if (fi_bpc != 8 && fi_bpc != 16 && fi_bpc != 32) {
            FLY_ERROR("FreeImage Error: Bits per channel not supported",
                     FLY_ERR_NOT_SUPPORTED);
        }

        // data type
        FREE_IMAGE_TYPE image_type = _.FreeImage_GetImageType(pBitmap.get());

        // sizes
        uint fi_w = _.FreeImage_GetWidth(pBitmap.get());
        uint fi_h = _.FreeImage_GetHeight(pBitmap.get());

        // FI = row major | FLY = column major
        uint nSrcPitch = _.FreeImage_GetPitch(pBitmap.get());
        const uchar* pSrcLine =
            _.FreeImage_GetBits(pBitmap.get()) + nSrcPitch * (fi_h - 1);

        // result image
        fly_array rImage;
        if (fi_color == 4) {  // 4 channel image
            if (fi_bpc == 8) {
                FLY_CHECK((readImage_t<uchar, FLYFI_RGBA>)(&rImage, pSrcLine,
                                                         nSrcPitch, fi_w,
                                                         fi_h));
            } else if (fi_bpc == 16) {
                FLY_CHECK((readImage_t<ushort, FLYFI_RGBA>)(&rImage, pSrcLine,
                                                          nSrcPitch, fi_w,
                                                          fi_h));
            } else if (fi_bpc == 32) {
                switch (image_type) {
                    case FIT_UINT32:
                        FLY_CHECK((readImage_t<uint, FLYFI_RGBA>)(&rImage,
                                                                pSrcLine,
                                                                nSrcPitch, fi_w,
                                                                fi_h));
                        break;
                    case FIT_INT32:
                        FLY_CHECK((readImage_t<int, FLYFI_RGBA>)(&rImage,
                                                               pSrcLine,
                                                               nSrcPitch, fi_w,
                                                               fi_h));
                        break;
                    case FIT_FLOAT:
                        FLY_CHECK((readImage_t<float, FLYFI_RGBA>)(&rImage,
                                                                 pSrcLine,
                                                                 nSrcPitch,
                                                                 fi_w, fi_h));
                        break;
                    default:
                        FLY_ERROR("FreeImage Error: Unknown image type",
                                 FLY_ERR_NOT_SUPPORTED);
                        break;
                }
            }
        } else if (fi_color == 1) {
            if (fi_bpc == 8) {
                FLY_CHECK((readImage_t<uchar, FLYFI_GRAY>)(&rImage, pSrcLine,
                                                         nSrcPitch, fi_w,
                                                         fi_h));
            } else if (fi_bpc == 16) {
                FLY_CHECK((readImage_t<ushort, FLYFI_GRAY>)(&rImage, pSrcLine,
                                                          nSrcPitch, fi_w,
                                                          fi_h));
            } else if (fi_bpc == 32) {
                switch (image_type) {
                    case FIT_UINT32:
                        FLY_CHECK((readImage_t<uint, FLYFI_GRAY>)(&rImage,
                                                                pSrcLine,
                                                                nSrcPitch, fi_w,
                                                                fi_h));
                        break;
                    case FIT_INT32:
                        FLY_CHECK((readImage_t<int, FLYFI_GRAY>)(&rImage,
                                                               pSrcLine,
                                                               nSrcPitch, fi_w,
                                                               fi_h));
                        break;
                    case FIT_FLOAT:
                        FLY_CHECK((readImage_t<float, FLYFI_GRAY>)(&rImage,
                                                                 pSrcLine,
                                                                 nSrcPitch,
                                                                 fi_w, fi_h));
                        break;
                    default:
                        FLY_ERROR("FreeImage Error: Unknown image type",
                                 FLY_ERR_NOT_SUPPORTED);
                        break;
                }
            }
        } else {  // 3 channel imag
            if (fi_bpc == 8) {
                FLY_CHECK((readImage_t<uchar, FLYFI_RGB>)(&rImage, pSrcLine,
                                                        nSrcPitch, fi_w, fi_h));
            } else if (fi_bpc == 16) {
                FLY_CHECK((readImage_t<ushort, FLYFI_RGB>)(&rImage, pSrcLine,
                                                         nSrcPitch, fi_w,
                                                         fi_h));
            } else if (fi_bpc == 32) {
                switch (image_type) {
                    case FIT_UINT32:
                        FLY_CHECK((readImage_t<uint, FLYFI_RGB>)(&rImage,
                                                               pSrcLine,
                                                               nSrcPitch, fi_w,
                                                               fi_h));
                        break;
                    case FIT_INT32:
                        FLY_CHECK((readImage_t<int, FLYFI_RGB>)(&rImage, pSrcLine,
                                                              nSrcPitch, fi_w,
                                                              fi_h));
                        break;
                    case FIT_FLOAT:
                        FLY_CHECK((readImage_t<float, FLYFI_RGB>)(&rImage,
                                                                pSrcLine,
                                                                nSrcPitch, fi_w,
                                                                fi_h));
                        break;
                    default:
                        FLY_ERROR("FreeImage Error: Unknown image type",
                                 FLY_ERR_NOT_SUPPORTED);
                        break;
                }
            }
        }

        std::swap(*out, rImage);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T, FI_CHANNELS channels>
static void save_t(T* pDstLine, const fly_array in, const dim4& dims,
                   uint nDstPitch) {
    fly_array rr = 0, gg = 0, bb = 0, aa = 0;
    FLY_CHECK(channel_split(in, dims, &rr, &gg, &bb,
                           &aa));  // convert array to 3 channels if needed

    fly_array rrT = 0, ggT = 0, bbT = 0, aaT = 0;
    T *pSrc0 = 0, *pSrc1 = 0, *pSrc2 = 0, *pSrc3 = 0;

    uint step = channels;  // force 3 channels saving
    uint indx = 0;

    FLY_CHECK(fly_transpose(&rrT, rr, false));
    if (channels >= 3) { FLY_CHECK(fly_transpose(&ggT, gg, false)); }
    if (channels >= 3) { FLY_CHECK(fly_transpose(&bbT, bb, false)); }
    if (channels >= 4) { FLY_CHECK(fly_transpose(&aaT, aa, false)); }

    const ArrayInfo& cinfo = getInfo(rrT);
    pSrc0                  = pinnedAlloc<T>(cinfo.elements());
    if (channels >= 3) { pSrc1 = pinnedAlloc<T>(cinfo.elements()); }
    if (channels >= 3) { pSrc2 = pinnedAlloc<T>(cinfo.elements()); }
    if (channels >= 4) { pSrc3 = pinnedAlloc<T>(cinfo.elements()); }

    FLY_CHECK(fly_get_data_ptr((void*)pSrc0, rrT));
    if (channels >= 3) { FLY_CHECK(fly_get_data_ptr((void*)pSrc1, ggT)); }
    if (channels >= 3) { FLY_CHECK(fly_get_data_ptr((void*)pSrc2, bbT)); }
    if (channels >= 4) { FLY_CHECK(fly_get_data_ptr((void*)pSrc3, aaT)); }

    const uint fi_w = dims[1];
    const uint fi_h = dims[0];

    // Copy the array into FreeImage buffer
    for (uint y = 0; y < fi_h; ++y) {
        for (uint x = 0; x < fi_w; ++x) {
            if (channels == 1) {
                *(pDstLine + x * step) = pSrc0[indx];  // r -> 0
            } else if (channels >= 3) {
                if (static_cast<fly_dtype>(fly::dtype_traits<T>::fly_type) == u8) {
                    *(pDstLine + x * step + FI_RGBA_RED) =
                        pSrc0[indx];  // r -> 0
                    *(pDstLine + x * step + FI_RGBA_GREEN) =
                        pSrc1[indx];  // g -> 1
                    *(pDstLine + x * step + FI_RGBA_BLUE) =
                        pSrc2[indx];  // b -> 2
                    if (channels >= 4) {
                        *(pDstLine + x * step + FI_RGBA_ALPHA) =
                            pSrc3[indx];  // a
                    }
                } else {
                    // Non 8-bit types do not use ordering
                    // See Pixel Access Functions Chapter in FreeImage Doc
                    *(pDstLine + x * step + 0) = pSrc0[indx];  // r -> 0
                    *(pDstLine + x * step + 1) = pSrc1[indx];  // g -> 1
                    *(pDstLine + x * step + 2) = pSrc2[indx];  // b -> 2
                    if (channels >= 4) {
                        *(pDstLine + x * step + 3) = pSrc3[indx];  // a
                    }
                }
            }
            ++indx;
        }
        pDstLine = reinterpret_cast<T*>(reinterpret_cast<uchar*>(pDstLine) -
                                        nDstPitch);
    }
    pinnedFree(pSrc0);
    if (channels >= 3) { pinnedFree(pSrc1); }
    if (channels >= 3) { pinnedFree(pSrc2); }
    if (channels >= 4) { pinnedFree(pSrc3); }

    if (rr != 0) { FLY_CHECK(fly_release_array(rr)); }
    if (gg != 0) { FLY_CHECK(fly_release_array(gg)); }
    if (bb != 0) { FLY_CHECK(fly_release_array(bb)); }
    if (aa != 0) { FLY_CHECK(fly_release_array(aa)); }
    if (rrT != 0) { FLY_CHECK(fly_release_array(rrT)); }
    if (ggT != 0) { FLY_CHECK(fly_release_array(ggT)); }
    if (bbT != 0) { FLY_CHECK(fly_release_array(bbT)); }
    if (aaT != 0) { FLY_CHECK(fly_release_array(aaT)); }
}

// Save an image to disk.
fly_err fly_save_image_native(const char* filename, const fly_array in) {
    try {
        ARG_ASSERT(0, filename != NULL);

        FreeImage_Module& _ = getFreeImagePlugin();

        // set your own FreeImage error handler
        _.FreeImage_SetOutputMessage(FreeImageErrorHandler);

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = _.FreeImage_GetFileType(filename, 0);
        if (fif == FIF_UNKNOWN) {
            fif = _.FreeImage_GetFIFFromFilename(filename);
        }

        if (fif == FIF_UNKNOWN) {
            FLY_ERROR("FreeImage Error: Unknown Filetype", FLY_ERR_NOT_SUPPORTED);
        }

        const ArrayInfo& info = getInfo(in);
        // check image color type
        auto channels = static_cast<FI_CHANNELS>(info.dims()[2]);
        DIM_ASSERT(1, channels <= 4);
        DIM_ASSERT(1, channels != 2);

        // sizes
        uint fi_w = info.dims()[1];
        uint fi_h = info.dims()[0];

        fly_dtype type = info.getType();

        // FI assumes [0-255] for u8
        // FI assumes [0-65k] for u16
        // FI assumes [0-1]   for f32
        int fi_bpp = 0;
        switch (type) {
            case u8: fi_bpp = channels * 8; break;
            case u16: fi_bpp = channels * 16; break;
            case f32: fi_bpp = channels * 32; break;
            default: TYPE_ERROR(1, type);
        }

        FREE_IMAGE_TYPE fit_type = getFIT(channels, type);

        // create the result image storage using FreeImage
        bitmap_ptr pResultBitmap = make_bitmap_ptr(nullptr);
        switch (type) {
            case u8:
            case u16:
            case f32:
                pResultBitmap.reset(_.FreeImage_AllocateT(fit_type, fi_w, fi_h,
                                                          fi_bpp, 0, 0, 0));
                break;
            default: TYPE_ERROR(1, type);
        }

        if (pResultBitmap == NULL) {
            FLY_ERROR("FreeImage Error: Error creating image or file",
                     FLY_ERR_RUNTIME);
        }

        // FI = row major | FLY = column major
        uint nDstPitch = _.FreeImage_GetPitch(pResultBitmap.get());
        void* pDstLine =
            _.FreeImage_GetBits(pResultBitmap.get()) + nDstPitch * (fi_h - 1);

        if (channels == FLYFI_GRAY) {
            switch (type) {
                case u8:
                    save_t<uchar, FLYFI_GRAY>(static_cast<uchar*>(pDstLine), in,
                                             info.dims(), nDstPitch);
                    break;
                case u16:
                    save_t<ushort, FLYFI_GRAY>(static_cast<ushort*>(pDstLine),
                                              in, info.dims(), nDstPitch);
                    break;
                case f32:
                    save_t<float, FLYFI_GRAY>(static_cast<float*>(pDstLine), in,
                                             info.dims(), nDstPitch);
                    break;
                default: TYPE_ERROR(1, type);
            }
        } else if (channels == FLYFI_RGB) {
            switch (type) {
                case u8:
                    save_t<uchar, FLYFI_RGB>(static_cast<uchar*>(pDstLine), in,
                                            info.dims(), nDstPitch);
                    break;
                case u16:
                    save_t<ushort, FLYFI_RGB>(static_cast<ushort*>(pDstLine), in,
                                             info.dims(), nDstPitch);
                    break;
                case f32:
                    save_t<float, FLYFI_RGB>(static_cast<float*>(pDstLine), in,
                                            info.dims(), nDstPitch);
                    break;
                default: TYPE_ERROR(1, type);
            }
        } else {
            switch (type) {
                case u8:
                    save_t<uchar, FLYFI_RGBA>(static_cast<uchar*>(pDstLine), in,
                                             info.dims(), nDstPitch);
                    break;
                case u16:
                    save_t<ushort, FLYFI_RGBA>(static_cast<ushort*>(pDstLine),
                                              in, info.dims(), nDstPitch);
                    break;
                case f32:
                    save_t<float, FLYFI_RGBA>(static_cast<float*>(pDstLine), in,
                                             info.dims(), nDstPitch);
                    break;
                default: TYPE_ERROR(1, type);
            }
        }

        unsigned flags = 0;
        if (fif == FIF_JPEG) {
            flags = flags | static_cast<unsigned>(JPEG_QUALITYSUPERB);
        }

        // now save the result image
        if (!(_.FreeImage_Save(fif, pResultBitmap.get(), filename,
                               static_cast<int>(flags)) == TRUE)) {
            FLY_ERROR("FreeImage Error: Failed to save image", FLY_ERR_RUNTIME);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_is_image_io_available(bool* out) {
    *out = true;
    return FLY_SUCCESS;
}

#else  // WITH_FREEIMAGE
#include <common/err_common.hpp>
#include <stdio.h>
#include <fly/image.h>
fly_err fly_load_image_native(fly_array* out, const char* filename) {
    FLY_RETURN_ERROR("Flare compiled without Image IO (FreeImage) support",
                    FLY_ERR_NOT_CONFIGURED);
}

fly_err fly_save_image_native(const char* filename, const fly_array in) {
    FLY_RETURN_ERROR("Flare compiled without Image IO (FreeImage) support",
                    FLY_ERR_NOT_CONFIGURED);
}

fly_err fly_is_image_io_available(bool* out) {
    *out = false;
    return FLY_SUCCESS;
}
#endif  // WITH_FREEIMAGE
