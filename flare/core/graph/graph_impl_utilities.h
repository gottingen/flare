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

#ifndef FLARE_CORE_GRAPH_GRAPH_IMPL_UTILITIES_H_
#define FLARE_CORE_GRAPH_GRAPH_IMPL_UTILITIES_H_

#include <flare/core/defines.h>

#include <flare/core/graph/graph_fwd.h>

#include <type_traits>

namespace flare::detail {

    template<class Src, class Dst, class Enable = void>
    struct is_compatible_type_erasure : std::false_type {
    };

    template<class T>
    struct is_compatible_type_erasure<T, flare::experimental::TypeErasedTag>
            : std::true_type {
    };

    template<>
    struct is_compatible_type_erasure<flare::experimental::TypeErasedTag,
            flare::experimental::TypeErasedTag>
            : std::true_type {
    };

    template<class T>
    struct is_compatible_type_erasure<T, T> : std::true_type {
    };

    // So there are a couple of ways we could do this, but I didn't want to set up
    // all of the machinery to do a lazy instantiation of the convertibility
    // condition in the converting constructor of GraphNodeRef, so I'm going with
    // this for now:
    // TODO @desul-integration make this variadic once we have a meta-conjunction
    template<template<class, class, class> class Template, class TSrc, class USrc,
            class VSrc, class TDst, class UDst, class VDst>
    struct is_compatible_type_erasure<
            Template<TSrc, USrc, VSrc>, Template<TDst, UDst, VDst>,
            // Because gcc thinks this is ambiguous, we need to add this:
            std::enable_if_t<!std::is_same<TSrc, TDst>::value ||
                             !std::is_same<USrc, UDst>::value ||
                             !std::is_same<VSrc, VDst>::value>>
            : std::bool_constant<is_compatible_type_erasure<TSrc, TDst>::value &&
                                 is_compatible_type_erasure<USrc, UDst>::value &&
                                 is_compatible_type_erasure<VSrc, VDst>::value> {
    };


    template<class T, class U>
    struct is_more_type_erased : std::false_type {
    };

    template<class T>
    struct is_more_type_erased<flare::experimental::TypeErasedTag, T>
            : std::true_type {
    };

    template<>
    struct is_more_type_erased<flare::experimental::TypeErasedTag,
            flare::experimental::TypeErasedTag>
            : std::false_type {
    };

// TODO @desul-integration variadic version of this, like the above


}  // end namespace flare::detail

#endif  // FLARE_CORE_GRAPH_GRAPH_IMPL_UTILITIES_H_
