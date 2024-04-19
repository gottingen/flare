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

#ifndef IMAGEIO_HELPER_H
#define IMAGEIO_HELPER_H

#include <common/DependencyModule.hpp>
#include <common/err_common.hpp>
#include <fly/array.h>
#include <fly/dim4.hpp>
#include <fly/index.h>

#include <FreeImage.h>

#include <functional>
#include <memory>

namespace flare {

class FreeImage_Module {
    common::DependencyModule module;

   public:
    MODULE_MEMBER(FreeImage_Allocate);
    MODULE_MEMBER(FreeImage_AllocateT);
    MODULE_MEMBER(FreeImage_CloseMemory);
    MODULE_MEMBER(FreeImage_DeInitialise);
    MODULE_MEMBER(FreeImage_FIFSupportsReading);
    MODULE_MEMBER(FreeImage_GetBPP);
    MODULE_MEMBER(FreeImage_GetBits);
    MODULE_MEMBER(FreeImage_GetColorType);
    MODULE_MEMBER(FreeImage_GetFIFFromFilename);
    MODULE_MEMBER(FreeImage_GetFileType);
    MODULE_MEMBER(FreeImage_GetFileTypeFromMemory);
    MODULE_MEMBER(FreeImage_GetHeight);
    MODULE_MEMBER(FreeImage_GetImageType);
    MODULE_MEMBER(FreeImage_GetPitch);
    MODULE_MEMBER(FreeImage_GetWidth);
    MODULE_MEMBER(FreeImage_Initialise);
    MODULE_MEMBER(FreeImage_Load);
    MODULE_MEMBER(FreeImage_LoadFromMemory);
    MODULE_MEMBER(FreeImage_OpenMemory);
    MODULE_MEMBER(FreeImage_Save);
    MODULE_MEMBER(FreeImage_SaveToMemory);
    MODULE_MEMBER(FreeImage_SeekMemory);
    MODULE_MEMBER(FreeImage_SetOutputMessage);
    MODULE_MEMBER(FreeImage_Unload);

    FreeImage_Module();
    ~FreeImage_Module();
};

FreeImage_Module &getFreeImagePlugin();

using bitmap_ptr = std::unique_ptr<FIBITMAP, std::function<void(FIBITMAP *)>>;
bitmap_ptr make_bitmap_ptr(FIBITMAP *);

typedef enum {
    FLYFI_GRAY = 1,  //< gray
    FLYFI_RGB  = 3,  //< rgb
    FLYFI_RGBA = 4   //< rgba
} FI_CHANNELS;

// Error handler for FreeImage library.
// In case this handler is invoked, it throws an fly exception.
static void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif,
                                  const char *zMessage) {
    UNUSED(oFif);
    printf("FreeImage Error Handler: %s\n", zMessage);
}

//  Split a MxNx3 image into 3 separate channel matrices.
//  Produce 3 channels if needed
static fly_err channel_split(const fly_array rgb, const fly::dim4 &dims,
                            fly_array *outr, fly_array *outg, fly_array *outb,
                            fly_array *outa) {
    try {
        fly_seq idx[4][3] = {{fly_span, fly_span, {0, 0, 1}},
                            {fly_span, fly_span, {1, 1, 1}},
                            {fly_span, fly_span, {2, 2, 1}},
                            {fly_span, fly_span, {3, 3, 1}}};

        if (dims[2] == 4) {
            FLY_CHECK(fly_index(outr, rgb, dims.ndims(), idx[0]));
            FLY_CHECK(fly_index(outg, rgb, dims.ndims(), idx[1]));
            FLY_CHECK(fly_index(outb, rgb, dims.ndims(), idx[2]));
            FLY_CHECK(fly_index(outa, rgb, dims.ndims(), idx[3]));
        } else if (dims[2] == 3) {
            FLY_CHECK(fly_index(outr, rgb, dims.ndims(), idx[0]));
            FLY_CHECK(fly_index(outg, rgb, dims.ndims(), idx[1]));
            FLY_CHECK(fly_index(outb, rgb, dims.ndims(), idx[2]));
        } else {
            FLY_CHECK(fly_index(outr, rgb, dims.ndims(), idx[0]));
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

#endif
}
