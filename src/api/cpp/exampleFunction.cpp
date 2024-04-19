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

#include <fly/array.h>  // fly::array class is declared here

#include <fly/util.h>  // Include the header related to the function

#include "error.hpp"  // FLY_THROW macro to use error code C-API
                      // is going to return and throw corresponding
                      // exceptions if call isn't a success

namespace fly {

array exampleFunction(const array& a, const fly_someenum_t p) {
    // create a temporary fly_array handle
    fly_array temp = 0;

    // call C-API function
    FLY_THROW(fly_example_function(&temp, a.get(), p));

    // array::get() returns fly_array handle for the corresponding cpp fly::array
    return array(temp);
}

}  // namespace fly
