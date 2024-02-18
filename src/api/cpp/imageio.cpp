/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/compatible.h>
#include <fly/image.h>
#include "error.hpp"

namespace fly {

array loadImage(const char* filename, const bool is_color) {
    fly_array out = 0;
    FLY_THROW(fly_load_image(&out, filename, is_color));
    return array(out);
}

array loadImageMem(const void* ptr) {
    fly_array out = 0;
    FLY_THROW(fly_load_image_memory(&out, ptr));
    return array(out);
}

array loadimage(const char* filename, const bool is_color) {
    return loadImage(filename, is_color);
}

void saveImage(const char* filename, const array& in) {
    FLY_THROW(fly_save_image(filename, in.get()));
}

void* saveImageMem(const array& in, const imageFormat format) {
    void* ptr = NULL;
    FLY_THROW(fly_save_image_memory(&ptr, in.get(), format));
    return ptr;
}

void saveimage(const char* filename, const array& in) {
    return saveImage(filename, in);
}

void deleteImageMem(void* ptr) { FLY_THROW(fly_delete_image_memory(ptr)); }

array loadImageNative(const char* filename) {
    fly_array out = 0;
    FLY_THROW(fly_load_image_native(&out, filename));
    return array(out);
}

void saveImageNative(const char* filename, const array& in) {
    FLY_THROW(fly_save_image_native(filename, in.get()));
}

bool isImageIOAvailable() {
    bool out = false;
    FLY_THROW(fly_is_image_io_available(&out));
    return out;
}

}  // namespace fly
