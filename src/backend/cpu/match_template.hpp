/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace flare {
namespace cpu {
template<typename inType, typename outType>
Array<outType> match_template(const Array<inType> &sImg,
                              const Array<inType> &tImg,
                              const fly::matchType mType);
}  // namespace cpu
}  // namespace flare
