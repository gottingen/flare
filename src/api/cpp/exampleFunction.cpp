/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
