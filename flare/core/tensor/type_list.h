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

#ifndef FLARE_CORE_TENSOR_TYPE_LIST_H_
#define FLARE_CORE_TENSOR_TYPE_LIST_H_

#include <flare/core/tensor/macros.h>
#include <flare/core/tensor/trait_backports.h> // make_index_sequence

namespace flare::detail {

    template<class... _Ts>
    struct __type_list {
        static constexpr auto __size = sizeof...(_Ts);
    };

// Implementation of type_list at() that's heavily optimized for small typelists
    template<size_t, class>
    struct __type_at;
    template<size_t, class _Seq, class=std::make_index_sequence<_Seq::__size>>
    struct __type_at_large_impl;

    template<size_t _I, size_t _Idx, class _T>
    struct __type_at_entry {
    };

    template<class _Result>
    struct __type_at_assign_op_ignore_rest {
        template<class _T>
        __type_at_assign_op_ignore_rest<_Result> operator=(_T &&);

        using type = _Result;
    };

    struct __type_at_assign_op_impl {
        template<size_t _I, size_t _Idx, class _T>
        __type_at_assign_op_impl operator=(__type_at_entry<_I, _Idx, _T> &&);

        template<size_t _I, class _T>
        __type_at_assign_op_ignore_rest<_T> operator=(__type_at_entry<_I, _I, _T> &&);
    };

    template<size_t _I, class... _Ts, size_t... _Idxs>
    struct __type_at_large_impl<_I, __type_list<_Ts...>, std::integer_sequence<size_t, _Idxs...>>
            : decltype(
              _MDSPAN_FOLD_ASSIGN_LEFT(__type_at_assign_op_impl{}, /* = ... = */ __type_at_entry<_I, _Idxs, _Ts>{})
              ) {
    };

    template<size_t _I, class... _Ts>
    struct __type_at<_I, __type_list<_Ts...>>
            : __type_at_large_impl<_I, __type_list<_Ts...>> {
    };

    template<class _T0, class... _Ts>
    struct __type_at<0, __type_list<_T0, _Ts...>> {
        using type = _T0;
    };

    template<class _T0, class _T1, class... _Ts>
    struct __type_at<1, __type_list<_T0, _T1, _Ts...>> {
        using type = _T1;
    };

    template<class _T0, class _T1, class _T2, class... _Ts>
    struct __type_at<2, __type_list<_T0, _T1, _T2, _Ts...>> {
        using type = _T2;
    };

    template<class _T0, class _T1, class _T2, class _T3, class... _Ts>
    struct __type_at<3, __type_list<_T0, _T1, _T2, _T3, _Ts...>> {
        using type = _T3;
    };

} // end namespace flare::detail

#endif  // FLARE_CORE_TENSOR_TYPE_LIST_H_
