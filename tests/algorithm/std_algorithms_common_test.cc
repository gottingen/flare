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

#include <std_algorithms_common_test.h>

namespace Test {
namespace stdalgos {

std::string view_tag_to_string(DynamicTag) { return "dynamic_view"; }

std::string view_tag_to_string(DynamicLayoutLeftTag) {
  return "dynamic_layout_left_view";
}

std::string view_tag_to_string(DynamicLayoutRightTag) {
  return "dynamic_layout_right_view";
}

std::string view_tag_to_string(StridedTwoTag) { return "stride2_view"; }

std::string view_tag_to_string(StridedThreeTag) { return "stride3_view"; }

std::string view_tag_to_string(StridedTwoRowsTag) { return "stride2rows_view"; }

std::string view_tag_to_string(StridedThreeRowsTag) {
  return "stride3rows_view";
}

}  // namespace stdalgos
}  // namespace Test
