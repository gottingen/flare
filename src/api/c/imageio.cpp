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

#include <common/DependencyModule.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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
using std::string;
using std::swap;

namespace flare {

template<typename T, FI_CHANNELS fi_color, FI_CHANNELS fo_color>
static fly_err readImage(fly_array* rImage, const uchar* pSrcLine,
                        const int nSrcPitch, const uint fi_w, const uint fi_h) {
    // create an array to receive the loaded image data.
    FLY_CHECK(fly_init());
    auto* pDst   = pinnedAlloc<float>(fi_w * fi_h * 4);  // 4 channels is max
    float* pDst0 = pDst;
    float* pDst1 = pDst + (fi_w * fi_h * 1);
    float* pDst2 = pDst + (fi_w * fi_h * 2);
    float* pDst3 = pDst + (fi_w * fi_h * 3);

    uint indx = 0;
    uint step = fi_color;

    for (uint x = 0; x < fi_w; ++x) {
        for (uint y = 0; y < fi_h; ++y) {
            const T* src = reinterpret_cast<const T*>(pSrcLine - y * nSrcPitch);
            if (fo_color == 1) {
                pDst0[indx] = static_cast<T>(*(src + (x * step)));
            } else if (fo_color >= 3) {
                if (static_cast<fly_dtype>(fly::dtype_traits<T>::fly_type) == u8) {
                    pDst0[indx] =
                        static_cast<float>(*(src + (x * step + FI_RGBA_RED)));
                    pDst1[indx] =
                        static_cast<float>(*(src + (x * step + FI_RGBA_GREEN)));
                    pDst2[indx] =
                        static_cast<float>(*(src + (x * step + FI_RGBA_BLUE)));
                    if (fo_color == 4) {
                        pDst3[indx] = static_cast<float>(
                            *(src + (x * step + FI_RGBA_ALPHA)));
                    }
                } else {
                    // Non 8-bit types do not use ordering
                    // See Pixel Access Functions Chapter in FreeImage Doc
                    pDst0[indx] = static_cast<float>(*(src + (x * step + 0)));
                    pDst1[indx] = static_cast<float>(*(src + (x * step + 1)));
                    pDst2[indx] = static_cast<float>(*(src + (x * step + 2)));
                    if (fo_color == 4) {
                        pDst3[indx] =
                            static_cast<float>(*(src + (x * step + 3)));
                    }
                }
            }
            indx++;
        }
    }

    fly::dim4 dims(fi_h, fi_w, fo_color, 1);
    fly_err err = fly_create_array(rImage, pDst, dims.ndims(), dims.get(),
                                 (fly_dtype)fly::dtype_traits<float>::fly_type);
    pinnedFree(pDst);
    return err;
}

#ifdef FREEIMAGE_STATIC
// NOTE: Redefine the MODULE_FUNCTION_INIT macro to call the static functions
// instead of dynamically loaded symbols in case we are building with a static
// FreeImage library
#undef MODULE_FUNCTION_INIT
#define MODULE_FUNCTION_INIT(NAME) NAME = &::NAME

FreeImage_Module::FreeImage_Module() : module(nullptr, nullptr) {
    // We don't care if the module loaded if we are staticly linking against
    // FreeImage
    ::FreeImage_Initialise(false);
#else
FreeImage_Module::FreeImage_Module() : module("freeimage", nullptr) {
    if (!module.isLoaded()) {
        string error_message =
            "Error loading FreeImage: " +
            common::DependencyModule::getErrorMessage() +
            "\nFreeImage or one of it's dependencies failed to "
            "load. Try installing FreeImage or check if FreeImage is in the "
            "search path.";
        FLY_ERROR(error_message.c_str(), FLY_ERR_LOAD_LIB);
    }
#endif
    MODULE_FUNCTION_INIT(FreeImage_Allocate);
    MODULE_FUNCTION_INIT(FreeImage_AllocateT);
    MODULE_FUNCTION_INIT(FreeImage_CloseMemory);
    MODULE_FUNCTION_INIT(FreeImage_DeInitialise);
    MODULE_FUNCTION_INIT(FreeImage_FIFSupportsReading);
    MODULE_FUNCTION_INIT(FreeImage_GetBPP);
    MODULE_FUNCTION_INIT(FreeImage_GetBits);
    MODULE_FUNCTION_INIT(FreeImage_GetColorType);
    MODULE_FUNCTION_INIT(FreeImage_GetFIFFromFilename);
    MODULE_FUNCTION_INIT(FreeImage_GetFileType);
    MODULE_FUNCTION_INIT(FreeImage_GetFileTypeFromMemory);
    MODULE_FUNCTION_INIT(FreeImage_GetHeight);
    MODULE_FUNCTION_INIT(FreeImage_GetImageType);
    MODULE_FUNCTION_INIT(FreeImage_GetPitch);
    MODULE_FUNCTION_INIT(FreeImage_GetWidth);
    MODULE_FUNCTION_INIT(FreeImage_Initialise);
    MODULE_FUNCTION_INIT(FreeImage_Load);
    MODULE_FUNCTION_INIT(FreeImage_LoadFromMemory);
    MODULE_FUNCTION_INIT(FreeImage_OpenMemory);
    MODULE_FUNCTION_INIT(FreeImage_Save);
    MODULE_FUNCTION_INIT(FreeImage_SaveToMemory);
    MODULE_FUNCTION_INIT(FreeImage_SeekMemory);
    MODULE_FUNCTION_INIT(FreeImage_SetOutputMessage);
    MODULE_FUNCTION_INIT(FreeImage_Unload);

#ifndef FREEIMAGE_STATIC
    if (!module.symbolsLoaded()) {
        string error_message =
            "Error loading FreeImage: " +
            common::DependencyModule::getErrorMessage() +
            "\nThe installed version of FreeImage is not compatible with "
            "Flare. Please create an issue on which this error message";
        FLY_ERROR(error_message.c_str(), FLY_ERR_LOAD_LIB);
    }
#endif
}

FreeImage_Module::~FreeImage_Module() {  // NOLINT(hicpp-use-equals-default,
                                         // modernize-use-equals-default)
#ifdef FREEIMAGE_STATIC
    getFreeImagePlugin().FreeImage_DeInitialise();
#endif
}

FreeImage_Module& getFreeImagePlugin() {
    static auto* plugin = new FreeImage_Module();
    return *plugin;
}

bitmap_ptr make_bitmap_ptr(FIBITMAP* ptr) {
    return bitmap_ptr(ptr, getFreeImagePlugin().FreeImage_Unload);
}

template<typename T, FI_CHANNELS fo_color>
static fly_err readImage(fly_array* rImage, const uchar* pSrcLine,
                        const int nSrcPitch, const uint fi_w, const uint fi_h) {
    // create an array to receive the loaded image data.
    FLY_CHECK(fly_init());
    auto* pDst = pinnedAlloc<float>(fi_w * fi_h);

    uint indx = 0;
    uint step = nSrcPitch / (fi_w * sizeof(T));
    T r, g, b;
    for (uint x = 0; x < fi_w; ++x) {
        for (uint y = 0; y < fi_h; ++y) {
            const T* src = reinterpret_cast<const T*>(pSrcLine - y * nSrcPitch);
            if (fo_color == 1) {
                pDst[indx] = static_cast<T>(*(src + (x * step)));
            } else if (fo_color >= 3) {
                if (static_cast<fly_dtype>(fly::dtype_traits<T>::fly_type) == u8) {
                    r = *(src + (x * step + FI_RGBA_RED));
                    g = *(src + (x * step + FI_RGBA_GREEN));
                    b = *(src + (x * step + FI_RGBA_BLUE));
                } else {
                    // Non 8-bit types do not use ordering
                    // See Pixel Access Functions Chapter in FreeImage Doc
                    r = *(src + (x * step + 0));
                    g = *(src + (x * step + 1));
                    b = *(src + (x * step + 2));
                }
                pDst[indx] = r * 0.2989f + g * 0.5870f + b * 0.1140f;
            }
            indx++;
        }
    }

    fly::dim4 dims(fi_h, fi_w, 1, 1);
    fly_err err = fly_create_array(rImage, pDst, dims.ndims(), dims.get(),
                                 (fly_dtype)fly::dtype_traits<float>::fly_type);
    pinnedFree(pDst);
    return err;
}

}  // namespace flare

////////////////////////////////////////////////////////////////////////////////
// File IO
////////////////////////////////////////////////////////////////////////////////
// Load image from disk.
fly_err fly_load_image(fly_array* out, const char* filename, const bool isColor) {
    using flare::readImage;
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
#ifdef JPEG_GREYSCALE
        if (fif == FIF_JPEG && !isColor) {
            flags = flags | static_cast<unsigned>(JPEG_GREYSCALE);
        }
#endif

        // check that the plugin has reading capabilities ...
        bitmap_ptr pBitmap = make_bitmap_ptr(NULL);
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
        if (isColor) {
            if (fi_color == 4) {  // 4 channel image
                if (fi_bpc == 8) {
                    FLY_CHECK((readImage<uchar, FLYFI_RGBA, FLYFI_RGBA>)(&rImage,
                                                                      pSrcLine,
                                                                      nSrcPitch,
                                                                      fi_w,
                                                                      fi_h));
                } else if (fi_bpc == 16) {
                    FLY_CHECK(
                        (readImage<ushort, FLYFI_RGBA, FLYFI_RGBA>)(&rImage,
                                                                  pSrcLine,
                                                                  nSrcPitch,
                                                                  fi_w, fi_h));
                } else if (fi_bpc == 32) {
                    switch (image_type) {
                        case FIT_UINT32:
                            FLY_CHECK((readImage<uint, FLYFI_RGBA,
                                                FLYFI_RGBA>)(&rImage, pSrcLine,
                                                            nSrcPitch, fi_w,
                                                            fi_h));
                            break;
                        case FIT_INT32:
                            FLY_CHECK((
                                readImage<int, FLYFI_RGBA, FLYFI_RGBA>)(&rImage,
                                                                      pSrcLine,
                                                                      nSrcPitch,
                                                                      fi_w,
                                                                      fi_h));
                            break;
                        case FIT_FLOAT:
                            FLY_CHECK((readImage<float, FLYFI_RGBA,
                                                FLYFI_RGBA>)(&rImage, pSrcLine,
                                                            nSrcPitch, fi_w,
                                                            fi_h));
                            break;
                        default:
                            FLY_ERROR("FreeImage Error: Unknown image type",
                                     FLY_ERR_NOT_SUPPORTED);
                            break;
                    }
                }
            } else if (fi_color == 1) {
                if (fi_bpc == 8) {
                    FLY_CHECK((readImage<uchar, FLYFI_GRAY, FLYFI_RGB>)(&rImage,
                                                                     pSrcLine,
                                                                     nSrcPitch,
                                                                     fi_w,
                                                                     fi_h));
                } else if (fi_bpc == 16) {
                    FLY_CHECK((readImage<ushort, FLYFI_GRAY, FLYFI_RGB>)(&rImage,
                                                                      pSrcLine,
                                                                      nSrcPitch,
                                                                      fi_w,
                                                                      fi_h));
                } else if (fi_bpc == 32) {
                    switch (image_type) {
                        case FIT_UINT32:
                            FLY_CHECK((
                                readImage<uint, FLYFI_GRAY, FLYFI_RGB>)(&rImage,
                                                                      pSrcLine,
                                                                      nSrcPitch,
                                                                      fi_w,
                                                                      fi_h));
                            break;
                        case FIT_INT32:
                            FLY_CHECK(
                                (readImage<int, FLYFI_GRAY, FLYFI_RGB>)(&rImage,
                                                                      pSrcLine,
                                                                      nSrcPitch,
                                                                      fi_w,
                                                                      fi_h));
                            break;
                        case FIT_FLOAT:
                            FLY_CHECK((readImage<float, FLYFI_GRAY,
                                                FLYFI_RGB>)(&rImage, pSrcLine,
                                                           nSrcPitch, fi_w,
                                                           fi_h));
                            break;
                        default:
                            FLY_ERROR("FreeImage Error: Unknown image type",
                                     FLY_ERR_NOT_SUPPORTED);
                            break;
                    }
                }
            } else {  // 3 channel image
                if (fi_bpc == 8) {
                    FLY_CHECK((
                        readImage<uchar, FLYFI_RGB, FLYFI_RGB>)(&rImage, pSrcLine,
                                                              nSrcPitch, fi_w,
                                                              fi_h));
                } else if (fi_bpc == 16) {
                    FLY_CHECK((readImage<ushort, FLYFI_RGB, FLYFI_RGB>)(&rImage,
                                                                     pSrcLine,
                                                                     nSrcPitch,
                                                                     fi_w,
                                                                     fi_h));
                } else if (fi_bpc == 32) {
                    switch (image_type) {
                        case FIT_UINT32:
                            FLY_CHECK(
                                (readImage<uint, FLYFI_RGB, FLYFI_RGB>)(&rImage,
                                                                      pSrcLine,
                                                                      nSrcPitch,
                                                                      fi_w,
                                                                      fi_h));
                            break;
                        case FIT_INT32:
                            FLY_CHECK(
                                (readImage<int, FLYFI_RGB, FLYFI_RGB>)(&rImage,
                                                                     pSrcLine,
                                                                     nSrcPitch,
                                                                     fi_w,
                                                                     fi_h));
                            break;
                        case FIT_FLOAT:
                            FLY_CHECK((
                                readImage<float, FLYFI_RGB, FLYFI_RGB>)(&rImage,
                                                                      pSrcLine,
                                                                      nSrcPitch,
                                                                      fi_w,
                                                                      fi_h));
                            break;
                        default:
                            FLY_ERROR("FreeImage Error: Unknown image type",
                                     FLY_ERR_NOT_SUPPORTED);
                            break;
                    }
                }
            }
        } else {                  // output gray irrespective
            if (fi_color == 1) {  // 4 channel image
                if (fi_bpc == 8) {
                    FLY_CHECK((readImage<uchar, FLYFI_GRAY>)(&rImage, pSrcLine,
                                                           nSrcPitch, fi_w,
                                                           fi_h));
                } else if (fi_bpc == 16) {
                    FLY_CHECK((readImage<ushort, FLYFI_GRAY>)(&rImage, pSrcLine,
                                                            nSrcPitch, fi_w,
                                                            fi_h));
                } else if (fi_bpc == 32) {
                    switch (image_type) {
                        case FIT_UINT32:
                            FLY_CHECK((readImage<uint, FLYFI_GRAY>)(&rImage,
                                                                  pSrcLine,
                                                                  nSrcPitch,
                                                                  fi_w, fi_h));
                            break;
                        case FIT_INT32:
                            FLY_CHECK((readImage<int, FLYFI_GRAY>)(&rImage,
                                                                 pSrcLine,
                                                                 nSrcPitch,
                                                                 fi_w, fi_h));
                            break;
                        case FIT_FLOAT:
                            FLY_CHECK((readImage<float, FLYFI_GRAY>)(&rImage,
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
            } else if (fi_color == 3 || fi_color == 4) {
                if (fi_bpc == 8) {
                    FLY_CHECK((readImage<uchar, FLYFI_RGB>)(&rImage, pSrcLine,
                                                          nSrcPitch, fi_w,
                                                          fi_h));
                } else if (fi_bpc == 16) {
                    FLY_CHECK((readImage<ushort, FLYFI_RGB>)(&rImage, pSrcLine,
                                                           nSrcPitch, fi_w,
                                                           fi_h));
                } else if (fi_bpc == 32) {
                    switch (image_type) {
                        case FIT_UINT32:
                            FLY_CHECK((readImage<uint, FLYFI_RGB>)(&rImage,
                                                                 pSrcLine,
                                                                 nSrcPitch,
                                                                 fi_w, fi_h));
                            break;
                        case FIT_INT32:
                            FLY_CHECK((readImage<int, FLYFI_RGB>)(&rImage,
                                                                pSrcLine,
                                                                nSrcPitch, fi_w,
                                                                fi_h));
                            break;
                        case FIT_FLOAT:
                            FLY_CHECK((readImage<float, FLYFI_RGB>)(&rImage,
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
            }
        }

        swap(*out, rImage);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

// Save an image to disk.
fly_err fly_save_image(const char* filename, const fly_array in_) {
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

        const ArrayInfo& info = getInfo(in_);
        // check image color type
        uint channels = info.dims()[2];
        DIM_ASSERT(1, channels <= 4);
        DIM_ASSERT(1, channels != 2);

        uint fi_bpp = channels * 8;

        // sizes
        uint fi_w = info.dims()[1];
        uint fi_h = info.dims()[0];

        // create the result image storage using FreeImage
        bitmap_ptr pResultBitmap = make_bitmap_ptr(_.FreeImage_Allocate(
            fi_w, fi_h, static_cast<int>(fi_bpp), 0, 0, 0));
        if (pResultBitmap == NULL) {
            FLY_ERROR("FreeImage Error: Error creating image or file",
                     FLY_ERR_RUNTIME);
        }

        // FI assumes [0-255]
        // If array is in 0-1 range, multiply by 255
        fly_array in;
        double max_real, max_imag;
        bool free_in = false;
        FLY_CHECK(fly_max_all(&max_real, &max_imag, in_));
        if (max_real <= 1) {
            fly_array c255 = 0;
            FLY_CHECK(fly_constant(&c255, 255.0, info.ndims(), info.dims().get(),
                                 f32));
            FLY_CHECK(fly_mul(&in, in_, c255, false));
            FLY_CHECK(fly_release_array(c255));
            free_in = true;
        } else if (max_real < 256) {  // NOLINT(bugprone-branch-clone)
            in = in_;
        } else if (max_real < 65536) {
            fly_array c255 = 0;
            FLY_CHECK(fly_constant(&c255, 257.0, info.ndims(), info.dims().get(),
                                 f32));
            FLY_CHECK(fly_div(&in, in_, c255, false));
            FLY_CHECK(fly_release_array(c255));
            free_in = true;
        } else {
            in = (in_);
        }

        // FI = row major | FLY = column major
        uint nDstPitch = _.FreeImage_GetPitch(pResultBitmap.get());
        uchar* pDstLine =
            _.FreeImage_GetBits(pResultBitmap.get()) + nDstPitch * (fi_h - 1);
        fly_array rr = 0, gg = 0, bb = 0, aa = 0;
        FLY_CHECK(channel_split(in, info.dims(), &rr, &gg, &bb,
                               &aa));  // convert array to 3 channels if needed

        uint step = channels;  // force 3 channels saving
        uint indx = 0;

        fly_array rrT = 0, ggT = 0, bbT = 0, aaT = 0;
        if (channels == 4) {
            FLY_CHECK(fly_transpose(&rrT, rr, false));
            FLY_CHECK(fly_transpose(&ggT, gg, false));
            FLY_CHECK(fly_transpose(&bbT, bb, false));
            FLY_CHECK(fly_transpose(&aaT, aa, false));

            const ArrayInfo& cinfo = getInfo(rrT);

            auto* pSrc0 = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc1 = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc2 = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc3 = pinnedAlloc<float>(cinfo.elements());

            FLY_CHECK(fly_get_data_ptr((void*)pSrc0, rrT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc1, ggT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc2, bbT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc3, aaT));

            // Copy the array into FreeImage buffer
            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step + FI_RGBA_RED) =
                        static_cast<uchar>(pSrc0[indx]);  // r
                    *(pDstLine + x * step + FI_RGBA_GREEN) =
                        static_cast<uchar>(pSrc1[indx]);  // g
                    *(pDstLine + x * step + FI_RGBA_BLUE) =
                        static_cast<uchar>(pSrc2[indx]);  // b
                    *(pDstLine + x * step + FI_RGBA_ALPHA) =
                        static_cast<uchar>(pSrc3[indx]);  // a
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
            pinnedFree(pSrc1);
            pinnedFree(pSrc2);
            pinnedFree(pSrc3);
        } else if (channels == 3) {
            FLY_CHECK(fly_transpose(&rrT, rr, false));
            FLY_CHECK(fly_transpose(&ggT, gg, false));
            FLY_CHECK(fly_transpose(&bbT, bb, false));

            const ArrayInfo& cinfo = getInfo(rrT);

            auto* pSrc0 = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc1 = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc2 = pinnedAlloc<float>(cinfo.elements());

            FLY_CHECK(fly_get_data_ptr((void*)pSrc0, rrT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc1, ggT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc2, bbT));

            // Copy the array into FreeImage buffer
            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step + FI_RGBA_RED) =
                        static_cast<uchar>(pSrc0[indx]);  // r
                    *(pDstLine + x * step + FI_RGBA_GREEN) =
                        static_cast<uchar>(pSrc1[indx]);  // g
                    *(pDstLine + x * step + FI_RGBA_BLUE) =
                        static_cast<uchar>(pSrc2[indx]);  // b
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
            pinnedFree(pSrc1);
            pinnedFree(pSrc2);
        } else {
            FLY_CHECK(fly_transpose(&rrT, rr, false));
            const ArrayInfo& cinfo = getInfo(rrT);
            auto* pSrc0            = pinnedAlloc<float>(cinfo.elements());
            FLY_CHECK(fly_get_data_ptr((void*)pSrc0, rrT));

            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step) = static_cast<uchar>(pSrc0[indx]);
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
        }

        unsigned flags = 0;
        if (fif == FIF_JPEG) {
            flags = flags | static_cast<unsigned>(JPEG_QUALITYSUPERB);
        }

        // now save the result image
        if (_.FreeImage_Save(fif, pResultBitmap.get(), filename,
                             static_cast<int>(flags)) == FALSE) {
            FLY_ERROR("FreeImage Error: Failed to save image", FLY_ERR_RUNTIME);
        }

        if (free_in) { FLY_CHECK(fly_release_array(in)); }
        if (rr != 0) { FLY_CHECK(fly_release_array(rr)); }
        if (gg != 0) { FLY_CHECK(fly_release_array(gg)); }
        if (bb != 0) { FLY_CHECK(fly_release_array(bb)); }
        if (aa != 0) { FLY_CHECK(fly_release_array(aa)); }
        if (rrT != 0) { FLY_CHECK(fly_release_array(rrT)); }
        if (ggT != 0) { FLY_CHECK(fly_release_array(ggT)); }
        if (bbT != 0) { FLY_CHECK(fly_release_array(bbT)); }
        if (aaT != 0) { FLY_CHECK(fly_release_array(aaT)); }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Memory IO
////////////////////////////////////////////////////////////////////////////////
/// Load image from memory.
fly_err fly_load_image_memory(fly_array* out, const void* ptr) {
    using flare::readImage;
    try {
        ARG_ASSERT(1, ptr != NULL);

        FreeImage_Module& _ = getFreeImagePlugin();

        // set your own FreeImage error handler
        _.FreeImage_SetOutputMessage(FreeImageErrorHandler);

        auto* stream = static_cast<FIMEMORY*>(const_cast<void*>(ptr));
        _.FreeImage_SeekMemory(stream, 0L, SEEK_SET);

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = _.FreeImage_GetFileTypeFromMemory(stream, 0);
        // if (fif == FIF_UNKNOWN) {
        //    fif = FreeImage_GetFIFFromFilenameFromMemory(filename);
        //}

        if (fif == FIF_UNKNOWN) {
            FLY_ERROR("FreeImage Error: Unknown File or Filetype",
                     FLY_ERR_NOT_SUPPORTED);
        }

        unsigned flags = 0;
        if (fif == FIF_JPEG) {
            flags = flags | static_cast<unsigned>(JPEG_ACCURATE);
        }

        // check that the plugin has reading capabilities ...
        bitmap_ptr pBitmap = make_bitmap_ptr(NULL);
        if (_.FreeImage_FIFSupportsReading(fif)) {
            pBitmap.reset(_.FreeImage_LoadFromMemory(fif, stream,
                                                     static_cast<int>(flags)));
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
        int fi_color;
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
                FLY_CHECK((readImage<uchar, FLYFI_RGBA, FLYFI_RGBA>)(&rImage,
                                                                  pSrcLine,
                                                                  nSrcPitch,
                                                                  fi_w, fi_h));
            } else if (fi_bpc == 16) {
                FLY_CHECK((readImage<ushort, FLYFI_RGBA, FLYFI_RGBA>)(&rImage,
                                                                   pSrcLine,
                                                                   nSrcPitch,
                                                                   fi_w, fi_h));
            } else if (fi_bpc == 32) {
                FLY_CHECK((readImage<float, FLYFI_RGBA, FLYFI_RGBA>)(&rImage,
                                                                  pSrcLine,
                                                                  nSrcPitch,
                                                                  fi_w, fi_h));
            }
        } else if (fi_color == 1) {  // 1 channel image
            if (fi_bpc == 8) {
                FLY_CHECK((readImage<uchar, FLYFI_GRAY>)(&rImage, pSrcLine,
                                                       nSrcPitch, fi_w, fi_h));
            } else if (fi_bpc == 16) {
                FLY_CHECK((readImage<ushort, FLYFI_GRAY>)(&rImage, pSrcLine,
                                                        nSrcPitch, fi_w, fi_h));
            } else if (fi_bpc == 32) {
                FLY_CHECK((readImage<float, FLYFI_GRAY>)(&rImage, pSrcLine,
                                                       nSrcPitch, fi_w, fi_h));
            }
        } else {  // 3 channel image
            if (fi_bpc == 8) {
                FLY_CHECK((readImage<uchar, FLYFI_RGB, FLYFI_RGB>)(&rImage,
                                                                pSrcLine,
                                                                nSrcPitch, fi_w,
                                                                fi_h));
            } else if (fi_bpc == 16) {
                FLY_CHECK((readImage<ushort, FLYFI_RGB, FLYFI_RGB>)(&rImage,
                                                                 pSrcLine,
                                                                 nSrcPitch,
                                                                 fi_w, fi_h));
            } else if (fi_bpc == 32) {
                FLY_CHECK((readImage<float, FLYFI_RGB, FLYFI_RGB>)(&rImage,
                                                                pSrcLine,
                                                                nSrcPitch, fi_w,
                                                                fi_h));
            }
        }

        swap(*out, rImage);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

// Save an image to memory.
fly_err fly_save_image_memory(void** ptr, const fly_array in_,
                            const fly_image_format format) {
    try {
        FreeImage_Module& _ = getFreeImagePlugin();

        // set our own FreeImage error handler
        _.FreeImage_SetOutputMessage(FreeImageErrorHandler);

        // try to guess the file format from the file extension
        auto fif = static_cast<FREE_IMAGE_FORMAT>(format);

        if (fif == FIF_UNKNOWN || fif > 34) {  // FreeImage FREE_IMAGE_FORMAT
                                               // has upto 34 enums as of 3.17
            FLY_ERROR("FreeImage Error: Unknown Filetype", FLY_ERR_NOT_SUPPORTED);
        }

        const ArrayInfo& info = getInfo(in_);
        // check image color type
        uint channels = info.dims()[2];
        DIM_ASSERT(1, channels <= 4);
        DIM_ASSERT(1, channels != 2);

        uint fi_bpp = channels * 8;

        // sizes
        uint fi_w = info.dims()[1];
        uint fi_h = info.dims()[0];

        // create the result image storage using FreeImage
        bitmap_ptr pResultBitmap = make_bitmap_ptr(_.FreeImage_Allocate(
            fi_w, fi_h, static_cast<int>(fi_bpp), 0, 0, 0));
        if (pResultBitmap == NULL) {
            FLY_ERROR("FreeImage Error: Error creating image or file",
                     FLY_ERR_RUNTIME);
        }

        // FI assumes [0-255]
        // If array is in 0-1 range, multiply by 255
        fly_array in;
        double max_real, max_imag;
        bool free_in = false;
        FLY_CHECK(fly_max_all(&max_real, &max_imag, in_));
        if (max_real <= 1) {
            fly_array c255;
            FLY_CHECK(fly_constant(&c255, 255.0, info.ndims(), info.dims().get(),
                                 f32));
            FLY_CHECK(fly_mul(&in, in_, c255, false));
            FLY_CHECK(fly_release_array(c255));
            free_in = true;
        } else {
            in = in_;
        }

        // FI = row major | FLY = column major
        uint nDstPitch = _.FreeImage_GetPitch(pResultBitmap.get());
        uchar* pDstLine =
            _.FreeImage_GetBits(pResultBitmap.get()) + nDstPitch * (fi_h - 1);
        fly_array rr = 0, gg = 0, bb = 0, aa = 0;
        FLY_CHECK(channel_split(in, info.dims(), &rr, &gg, &bb,
                               &aa));  // convert array to 3 channels if needed

        uint step = channels;  // force 3 channels saving
        uint indx = 0;

        fly_array rrT = 0, ggT = 0, bbT = 0, aaT = 0;
        if (channels == 4) {
            FLY_CHECK(fly_transpose(&rrT, rr, false));
            FLY_CHECK(fly_transpose(&ggT, gg, false));
            FLY_CHECK(fly_transpose(&bbT, bb, false));
            FLY_CHECK(fly_transpose(&aaT, aa, false));

            const ArrayInfo& cinfo = getInfo(rrT);
            auto* pSrc0            = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc1            = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc2            = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc3            = pinnedAlloc<float>(cinfo.elements());

            FLY_CHECK(fly_get_data_ptr((void*)pSrc0, rrT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc1, ggT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc2, bbT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc3, aaT));

            // Copy the array into FreeImage buffer
            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step + FI_RGBA_RED) =
                        static_cast<uchar>(pSrc0[indx]);  // r
                    *(pDstLine + x * step + FI_RGBA_GREEN) =
                        static_cast<uchar>(pSrc1[indx]);  // g
                    *(pDstLine + x * step + FI_RGBA_BLUE) =
                        static_cast<uchar>(pSrc2[indx]);  // b
                    *(pDstLine + x * step + FI_RGBA_ALPHA) =
                        static_cast<uchar>(pSrc3[indx]);  // a
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
            pinnedFree(pSrc1);
            pinnedFree(pSrc2);
            pinnedFree(pSrc3);
        } else if (channels == 3) {
            FLY_CHECK(fly_transpose(&rrT, rr, false));
            FLY_CHECK(fly_transpose(&ggT, gg, false));
            FLY_CHECK(fly_transpose(&bbT, bb, false));

            const ArrayInfo& cinfo = getInfo(rrT);
            auto* pSrc0            = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc1            = pinnedAlloc<float>(cinfo.elements());
            auto* pSrc2            = pinnedAlloc<float>(cinfo.elements());

            FLY_CHECK(fly_get_data_ptr((void*)pSrc0, rrT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc1, ggT));
            FLY_CHECK(fly_get_data_ptr((void*)pSrc2, bbT));

            // Copy the array into FreeImage buffer
            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step + FI_RGBA_RED) =
                        static_cast<uchar>(pSrc0[indx]);  // r
                    *(pDstLine + x * step + FI_RGBA_GREEN) =
                        static_cast<uchar>(pSrc1[indx]);  // g
                    *(pDstLine + x * step + FI_RGBA_BLUE) =
                        static_cast<uchar>(pSrc2[indx]);  // b
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
            pinnedFree(pSrc1);
            pinnedFree(pSrc2);
        } else {
            FLY_CHECK(fly_transpose(&rrT, rr, false));
            const ArrayInfo& cinfo = getInfo(rrT);
            auto* pSrc0            = pinnedAlloc<float>(cinfo.elements());
            FLY_CHECK(fly_get_data_ptr((void*)pSrc0, rrT));

            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step) = static_cast<uchar>(pSrc0[indx]);
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
        }

        uint8_t* data          = nullptr;
        uint32_t size_in_bytes = 0;
        FIMEMORY* stream       = _.FreeImage_OpenMemory(data, size_in_bytes);

        unsigned flags = 0;
        if (fif == FIF_JPEG) {
            flags = flags | static_cast<unsigned>(JPEG_QUALITYSUPERB);
        }

        // now save the result image
        if (_.FreeImage_SaveToMemory(fif, pResultBitmap.get(), stream,
                                     static_cast<int>(flags)) == FALSE) {
            FLY_ERROR("FreeImage Error: Failed to save image", FLY_ERR_RUNTIME);
        }

        *ptr = stream;

        if (free_in) { FLY_CHECK(fly_release_array(in)); }
        if (rr != 0) { FLY_CHECK(fly_release_array(rr)); }
        if (gg != 0) { FLY_CHECK(fly_release_array(gg)); }
        if (bb != 0) { FLY_CHECK(fly_release_array(bb)); }
        if (aa != 0) { FLY_CHECK(fly_release_array(aa)); }
        if (rrT != 0) { FLY_CHECK(fly_release_array(rrT)); }
        if (ggT != 0) { FLY_CHECK(fly_release_array(ggT)); }
        if (bbT != 0) { FLY_CHECK(fly_release_array(bbT)); }
        if (aaT != 0) { FLY_CHECK(fly_release_array(aaT)); }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_delete_image_memory(void* ptr) {
    try {
        ARG_ASSERT(0, ptr != NULL);

        FreeImage_Module& _ = getFreeImagePlugin();

        // set your own FreeImage error handler
        _.FreeImage_SetOutputMessage(FreeImageErrorHandler);

        auto* stream = static_cast<FIMEMORY*>(ptr);
        _.FreeImage_SeekMemory(stream, 0L, SEEK_SET);

        // Ensure data is freeimage compatible
        FREE_IMAGE_FORMAT fif =
            _.FreeImage_GetFileTypeFromMemory(static_cast<FIMEMORY*>(ptr), 0);
        if (fif == FIF_UNKNOWN) {
            FLY_ERROR("FreeImage Error: Unknown Filetype", FLY_ERR_NOT_SUPPORTED);
        }

        _.FreeImage_CloseMemory(static_cast<FIMEMORY*>(ptr));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

#else  // WITH_FREEIMAGE
#include <common/err_common.hpp>
#include <stdio.h>
#include <fly/image.h>
fly_err fly_load_image(fly_array *out, const char *filename, const bool isColor) {
    FLY_RETURN_ERROR("Flare compiled without Image IO (FreeImage) support",
                    FLY_ERR_NOT_CONFIGURED);
}

fly_err fly_save_image(const char *filename, const fly_array in_) {
    FLY_RETURN_ERROR("Flare compiled without Image IO (FreeImage) support",
                    FLY_ERR_NOT_CONFIGURED);
}

fly_err fly_load_image_memory(fly_array *out, const void *ptr) {
    FLY_RETURN_ERROR("Flare compiled without Image IO (FreeImage) support",
                    FLY_ERR_NOT_CONFIGURED);
}

fly_err fly_save_image_memory(void **ptr, const fly_array in_,
                            const fly_image_format format) {
    FLY_RETURN_ERROR("Flare compiled without Image IO (FreeImage) support",
                    FLY_ERR_NOT_CONFIGURED);
}

fly_err fly_delete_image_memory(void *ptr) {
    FLY_RETURN_ERROR("Flare compiled without Image IO (FreeImage) support",
                    FLY_ERR_NOT_CONFIGURED);
}
#endif  // WITH_FREEIMAGE
