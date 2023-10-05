// Copyright 2023 The Elastic-AI Authors.
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

#ifndef FLARE_TEST_SUBVIEW_C04_HPP
#define FLARE_TEST_SUBVIEW_C04_HPP

#include <view/view_subview_test.h>

namespace Test {

    TEST_CASE("TEST_CATEGORY, view_subview_2d_from_3d") {
        TestViewSubview::test_2d_subview_3d<TEST_EXECSPACE>();
    }

}  // namespace Test
#endif
