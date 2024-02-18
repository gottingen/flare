/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/util.h>
#include <cstdio>
#include "error.hpp"

using namespace std;

namespace fly {
void print(const char *exp, const array &arr) {
    FLY_THROW(fly_print_array_gen(exp, arr.get(), 4));
}

void print(const char *exp, const array &arr, const int precision) {
    FLY_THROW(fly_print_array_gen(exp, arr.get(), precision));
}

int saveArray(const char *key, const array &arr, const char *filename,
              const bool append) {
    int index = -1;
    FLY_THROW(fly_save_array(&index, key, arr.get(), filename, append));
    return index;
}

array readArray(const char *filename, const unsigned index) {
    fly_array out = 0;
    FLY_THROW(fly_read_array_index(&out, filename, index));
    return array(out);
}

array readArray(const char *filename, const char *key) {
    fly_array out = 0;
    FLY_THROW(fly_read_array_key(&out, filename, key));
    return array(out);
}

int readArrayCheck(const char *filename, const char *key) {
    int out = -1;
    FLY_THROW(fly_read_array_key_check(&out, filename, key));
    return out;
}

void toString(char **output, const char *exp, const array &arr,
              const int precision, const bool transpose) {
    FLY_THROW(fly_array_to_string(output, exp, arr.get(), precision, transpose));
}

const char *toString(const char *exp, const array &arr, const int precision,
                     const bool transpose) {
    char *output = NULL;
    FLY_THROW(fly_array_to_string(&output, exp, arr.get(), precision, transpose));
    return output;
}

size_t getSizeOf(fly::dtype type) {
    size_t size = 0;
    FLY_THROW(fly_get_size_of(&size, type));
    return size;
}
}  // namespace fly
