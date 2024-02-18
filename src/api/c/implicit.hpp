/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/cast.hpp>
#include <handle.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <fly/array.h>
#include <fly/defines.h>

fly_dtype implicit(const fly_array lhs, const fly_array rhs);
fly_dtype implicit(const fly_dtype lty, const fly_dtype rty);
