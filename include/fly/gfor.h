/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fly/defines.h>
#include <fly/seq.h>


#ifdef __cplusplus
namespace fly
{
class array;
class dim4;

FLY_API bool gforToggle();
FLY_API bool gforGet();
FLY_API void gforSet(bool val);


#define gfor(var, ...) for (var = fly::seq(fly::seq(__VA_ARGS__), true); fly::gforToggle(); )

typedef array (*batchFunc_t)(const array &lhs, const array &rhs);
FLY_API array batchFunc(const array &lhs, const array &rhs, batchFunc_t func);

}
#endif
