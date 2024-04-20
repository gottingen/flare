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
