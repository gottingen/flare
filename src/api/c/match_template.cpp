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

#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <match_template.hpp>
#include <types.hpp>
#include <fly/defines.h>
#include <fly/vision.h>

#include <type_traits>

using fly::dim4;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::conditional;
using std::is_same;

template<typename InType>
static fly_array match_template(const fly_array& sImg, const fly_array tImg,
                               fly_match_type mType) {
    using OutType = typename conditional<is_same<InType, double>::value, double,
                                         float>::type;
    return getHandle(match_template<InType, OutType>(
        getArray<InType>(sImg), getArray<InType>(tImg), mType));
}

fly_err fly_match_template(fly_array* out, const fly_array search_img,
                         const fly_array template_img,
                         const fly_match_type m_type) {
    try {
        ARG_ASSERT(3, (m_type >= FLY_SAD && m_type <= FLY_LSSD));

        const ArrayInfo& sInfo = getInfo(search_img);
        const ArrayInfo& tInfo = getInfo(template_img);

        dim4 const& sDims = sInfo.dims();
        dim4 const& tDims = tInfo.dims();

        dim_t sNumDims = sDims.ndims();
        dim_t tNumDims = tDims.ndims();
        ARG_ASSERT(1, (sNumDims >= 2));
        ARG_ASSERT(2, (tNumDims == 2));

        fly_dtype sType = sInfo.getType();
        ARG_ASSERT(1, (sType == tInfo.getType()));

        fly_array output = 0;
        switch (sType) {
            case f64:
                output =
                    match_template<double>(search_img, template_img, m_type);
                break;
            case f32:
                output =
                    match_template<float>(search_img, template_img, m_type);
                break;
            case s32:
                output = match_template<int>(search_img, template_img, m_type);
                break;
            case u32:
                output = match_template<uint>(search_img, template_img, m_type);
                break;
            case s16:
                output =
                    match_template<short>(search_img, template_img, m_type);
                break;
            case u16:
                output =
                    match_template<ushort>(search_img, template_img, m_type);
                break;
            case b8:
                output = match_template<char>(search_img, template_img, m_type);
                break;
            case u8:
                output =
                    match_template<uchar>(search_img, template_img, m_type);
                break;
            default: TYPE_ERROR(1, sType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
