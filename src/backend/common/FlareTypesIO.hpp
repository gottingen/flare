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
#include <common/Version.hpp>
#include <collie/strings/format.h>
#include <fly/dim4.hpp>
#include <fly/seq.h>
#include <complex>

template<>
struct fmt::formatter<fly_seq> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const fly_seq& p, FormatContext& ctx) -> decltype(ctx.out()) {
        // ctx.out() is an output iterator to write to.
        if (p.begin == fly_span.begin && p.end == fly_span.end &&
            p.step == fly_span.step) {
            return format_to(ctx.out(), "span");
        }
        if (p.begin == p.end) { return format_to(ctx.out(), "{}", p.begin); }
        if (p.step == 1) {
            return format_to(ctx.out(), "({} -> {})", p.begin, p.end);
        }
        return format_to(ctx.out(), "({} -({})-> {})", p.begin, p.step, p.end);
    }
};

#if FMT_VERSION >= 90000
template<>
struct fmt::formatter<fly::dim4> : ostream_formatter {};
template<>
struct fmt::formatter<std::complex<float>> : ostream_formatter {};
template<>
struct fmt::formatter<std::complex<double>> : ostream_formatter {};
#endif

template<>
struct fmt::formatter<flare::common::Version> {
    // show major version
    bool show_major = false;
    // show minor version
    bool show_minor = false;
    // show patch version
    bool show_patch = false;

    // Parses format specifications of the form ['M' | 'm' | 'p'].
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it == end || *it == '}') {
            show_major = show_minor = show_patch = true;
            return it;
        }
        do {
            switch (*it) {
                case 'M': show_major = true; break;
                case 'm': show_minor = true; break;
                case 'p': show_patch = true; break;
                default: throw format_error("invalid format");
            }
            ++it;
        } while (it != end && *it != '}');
        return it;
    }

    template<typename FormatContext>
    auto format(const flare::common::Version& ver, FormatContext& ctx)
        -> decltype(ctx.out()) {
        if (ver.major() == -1) return format_to(ctx.out(), "N/A");
        if (ver.minor() == -1) show_minor = false;
        if (ver.patch() == -1) show_patch = false;
        if (show_major && !show_minor && !show_patch) {
            return format_to(ctx.out(), "{}", ver.major());
        }
        if (show_major && show_minor && !show_patch) {
            return format_to(ctx.out(), "{}.{}", ver.major(), ver.minor());
        }
        if (show_major && show_minor && show_patch) {
            return format_to(ctx.out(), "{}.{}.{}", ver.major(), ver.minor(),
                             ver.patch());
        }
        return ctx.out();
    }
};
