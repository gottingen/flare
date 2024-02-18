/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once
#include <fly/array.h>
#include <fly/features.h>
#include <cstddef>

typedef struct {
    size_t n;
    fly_array x;
    fly_array y;
    fly_array score;
    fly_array orientation;
    fly_array size;
} fly_features_t;

fly_features getFeaturesHandle(const fly_features_t feat);

fly_features_t getFeatures(const fly_features featHandle);
