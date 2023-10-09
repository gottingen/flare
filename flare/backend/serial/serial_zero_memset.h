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

#ifndef FLARE_BACKEND_SERIAL_SERIAL_ZERO_MEMSET_H_
#define FLARE_BACKEND_SERIAL_SERIAL_ZERO_MEMSET_H_

#include <flare/core/defines.h>
#include <flare/core/common/zero_memset_fwd.h>
#include <flare/backend/serial/serial.h>

#include <type_traits>

namespace flare::detail {

    // We only need to provide a specialization for Serial if there is a host
    // parallel execution space since the specialization for
    // DefaultHostExecutionSpace is defined elsewhere.
    struct DummyExecutionSpace;

    template<class T, class... P>
    struct ZeroMemset<
            std::conditional_t<!std::is_same<Serial, DefaultHostExecutionSpace>::value,
                    Serial, DummyExecutionSpace>,
            View<T, P...>>
            : public ZeroMemset<DefaultHostExecutionSpace, View<T, P...>> {
        using Base = ZeroMemset<DefaultHostExecutionSpace, View<T, P...>>;
        using Base::Base;

        ZeroMemset(const Serial &, const View<T, P...> &dst,
                   typename View<T, P...>::const_value_type &value)
                : Base(dst, value) {}
    };

}  // namespace flare::detail

#endif  // FLARE_BACKEND_SERIAL_SERIAL_ZERO_MEMSET_H_
