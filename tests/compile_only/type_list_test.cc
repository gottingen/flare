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

#include <flare/core/common/utilities.h>

using TypeList2 = flare::detail::type_list<void, bool>;
using TypeList3 = flare::detail::type_list<char, short, int>;
using TypeList223 =
        flare::detail::type_list<void, bool, void, bool, char, short, int>;
using TypeList223Void = flare::detail::type_list<void, void>;
using TypeList223NoVoid = flare::detail::type_list<bool, bool, char, short, int>;

// concat_type_list
using ConcatTypeList2 = flare::detail::concat_type_list_t<TypeList2>;
static_assert(std::is_same<TypeList2, ConcatTypeList2>::value,
              "concat_type_list of a single type_list failed");

using ConcatTypeList223 =
        flare::detail::concat_type_list_t<TypeList2, TypeList2, TypeList3>;
static_assert(std::is_same<TypeList223, ConcatTypeList223>::value,
              "concat_type_list of three type_lists failed");

// filter_type_list
using FilterTypeList223Void =
        flare::detail::filter_type_list_t<std::is_void, TypeList223>;
static_assert(std::is_same<TypeList223Void, FilterTypeList223Void>::value,
              "filter_type_list with predicate value==true failed");

using FilterTypeList223NoVoid =
        flare::detail::filter_type_list_t<std::is_void, TypeList223, false>;
static_assert(std::is_same<TypeList223NoVoid, FilterTypeList223NoVoid>::value,
              "filter_type_list with predicate value==false failed");
