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
