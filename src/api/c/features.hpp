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
