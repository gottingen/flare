/*******************************************************
 * Copyright (c) 2015, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/util.h>
#include "symbol_manager.hpp"

fly_err fly_print_array(fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_print_array, arr);
}

fly_err fly_print_array_gen(const char *exp, const fly_array arr,
                          const int precision) {
    CHECK_ARRAYS(arr);
    CALL(fly_print_array_gen, exp, arr, precision);
}

fly_err fly_save_array(int *index, const char *key, const fly_array arr,
                     const char *filename, const bool append) {
    CHECK_ARRAYS(arr);
    CALL(fly_save_array, index, key, arr, filename, append);
}

fly_err fly_read_array_index(fly_array *out, const char *filename,
                           const unsigned index) {
    CALL(fly_read_array_index, out, filename, index);
}

fly_err fly_read_array_key(fly_array *out, const char *filename, const char *key) {
    CALL(fly_read_array_key, out, filename, key);
}

fly_err fly_read_array_key_check(int *index, const char *filename,
                               const char *key) {
    CALL(fly_read_array_key_check, index, filename, key);
}

fly_err fly_array_to_string(char **output, const char *exp, const fly_array arr,
                          const int precision, const bool transpose) {
    CHECK_ARRAYS(arr);
    CALL(fly_array_to_string, output, exp, arr, precision, transpose);
}

fly_err fly_example_function(fly_array *out, const fly_array a,
                           const fly_someenum_t param) {
    CHECK_ARRAYS(a);
    CALL(fly_example_function, out, a, param);
}
