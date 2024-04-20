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

/// This file contains platform independent utility functions
#if defined(OS_WIN)
#include <Windows.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

#include <common/Logger.hpp>
#include <common/TemplateArg.hpp>
#include <common/defines.hpp>
#include <common/util.hpp>
#include <optypes.hpp>
#include <fly/defines.h>

#include <fly/span.hpp>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __has_include
#if __has_include(<charconv>)
#include <charconv>
#endif
#if __has_include(<version>)
#include <version>
#endif
#endif

using nonstd::span;
using std::accumulate;
using std::array;
using std::hash;
using std::ofstream;
using std::once_flag;
using std::rename;
using std::size_t;
using std::string;
using std::stringstream;
using std::thread;
using std::to_string;
using std::uint8_t;
using std::vector;

namespace flare {
namespace common {
// http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring/217605#217605
// trim from start
string& ltrim(string& s) {
    s.erase(s.begin(),
            find_if(s.begin(), s.end(), [](char c) { return !isspace(c); }));
    return s;
}

string getEnvVar(const string& key) {
#if defined(OS_WIN)
    DWORD bufSize =
        32767;  // limit according to GetEnvironment Variable documentation
    string retVal;
    retVal.resize(bufSize);
    bufSize = GetEnvironmentVariable(key.c_str(), &retVal[0], bufSize);
    if (!bufSize) {
        return string("");
    } else {
        retVal.resize(bufSize);
        return retVal;
    }
#else
    char* str = getenv(key.c_str());
    return str == NULL ? string("") : string(str);
#endif
}

const char* getName(fly_dtype type) {
    switch (type) {
        case f32: return "float";
        case f64: return "double";
        case c32: return "complex float";
        case c64: return "complex double";
        case u32: return "unsigned int";
        case s32: return "int";
        case u16: return "unsigned short";
        case s16: return "short";
        case u64: return "unsigned long long";
        case s64: return "long long";
        case u8: return "unsigned char";
        case b8: return "bool";
        default: return "unknown type";
    }
}

void saveKernel(const string& funcName, const string& jit_ker,
                const string& ext) {
    static constexpr const char* saveJitKernelsEnvVarName =
        "FLY_JIT_KERNEL_TRACE";
    static const char* jitKernelsOutput = getenv(saveJitKernelsEnvVarName);
    if (!jitKernelsOutput) { return; }
    if (strcmp(jitKernelsOutput, "stdout") == 0) {
        fputs(jit_ker.c_str(), stdout);
        return;
    }
    if (strcmp(jitKernelsOutput, "stderr") == 0) {
        fputs(jit_ker.c_str(), stderr);
        return;
    }
    // Path to a folder
    const string ffp =
        string(jitKernelsOutput) + FLY_PATH_SEPARATOR + funcName + ext;
    FILE* f = fopen(ffp.c_str(), "we");
    if (!f) {
        fprintf(stderr, "Cannot open file %s\n", ffp.c_str());
        return;
    }
    if (fputs(jit_ker.c_str(), f) == EOF) {
        fprintf(stderr, "Failed to write kernel to file %s\n", ffp.c_str());
    }
    fclose(f);
}

#if defined(OS_WIN)
string getTemporaryDirectory() {
    DWORD bufSize = 261;  // limit according to GetTempPath documentation
    string retVal;
    retVal.resize(bufSize);
    bufSize = GetTempPathA(bufSize, &retVal[0]);
    retVal.resize(bufSize);
    return retVal;
}
#else
string getHomeDirectory() {
    string home = getEnvVar("XDG_CACHE_HOME");
    if (!home.empty()) { return home; }

    home = getEnvVar("HOME");
    if (!home.empty()) { return home; }

    return getpwuid(getuid())->pw_dir;
}
#endif

bool directoryExists(const string& path) {
#if defined(OS_WIN)
    struct _stat status;
    return _stat(path.c_str(), &status) == 0 && (status.st_mode & S_IFDIR) != 0;
#else
    struct stat status {};
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    return stat(path.c_str(), &status) == 0 && (status.st_mode & S_IFDIR) != 0;
#endif
}

bool createDirectory(const string& path) {
#if defined(OS_WIN)
    return CreateDirectoryA(path.c_str(), NULL) != 0;
#else
    return mkdir(path.c_str(), 0777) == 0;
#endif
}

bool removeFile(const string& path) {
#if defined(OS_WIN)
    return DeleteFileA(path.c_str()) != 0;
#else
    return unlink(path.c_str()) == 0;
#endif
}

bool renameFile(const string& sourcePath, const string& destPath) {
    return rename(sourcePath.c_str(), destPath.c_str()) == 0;
}

bool isDirectoryWritable(const string& path) {
    if (!directoryExists(path) && !createDirectory(path)) { return false; }

    const string testPath = path + FLY_PATH_SEPARATOR + "test";
    if (!ofstream(testPath).is_open()) { return false; }
    removeFile(testPath);

    return true;
}

#ifndef NO_COLLIE_LOG
string& getCacheDirectory() {
    static once_flag flag;
    static string cacheDirectory;

    call_once(flag, []() {
        string pathList[] = {
#if defined(OS_WIN)
            getTemporaryDirectory() + "\\Flare"
#else
            getHomeDirectory() + "/.flare",
            "/tmp/flare"
#endif
        };

        auto env_path = getEnvVar(JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME);
        if (!env_path.empty() && !isDirectoryWritable(env_path)) {
            clog::get("platform")
                ->warn(
                    "The environment variable {}({}) is "
                    "not writeable. Falling back to default.",
                    JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME, env_path);
            env_path.clear();
        }

        if (env_path.empty()) {
            auto iterDir =
                find_if(begin(pathList), end(pathList), isDirectoryWritable);

            cacheDirectory = iterDir != end(pathList) ? *iterDir : "";
        } else {
            cacheDirectory = env_path;
        }
    });

    return cacheDirectory;
}
#endif

string makeTempFilename() {
    thread_local size_t fileCount = 0u;

    ++fileCount;
    const size_t threadID = hash<thread::id>{}(std::this_thread::get_id());

    return to_string(
        hash<string>{}(to_string(threadID) + "_" + to_string(fileCount)));
}

template<typename T>
string toString(T value) {
#ifdef __cpp_lib_to_chars
    array<char, 128> out;
    if (auto [ptr, ec] = std::to_chars(out.data(), out.data() + 128, value);
        ec == std::errc()) {
        return string(out.data(), ptr);
    } else {
        return string("#error invalid conversion");
    }
#else
    stringstream ss;
    ss.imbue(std::locale::classic());
    ss << value;
    return ss.str();
#endif
}

template string toString<int>(int);
template string toString<unsigned short>(unsigned short);
template string toString<short>(short);
template string toString<unsigned char>(unsigned char);
template string toString<char>(char);
template string toString<long>(long);
template string toString<long long>(long long);
template string toString<unsigned>(unsigned);
template string toString<unsigned long>(unsigned long);
template string toString<unsigned long long>(unsigned long long);
template string toString<float>(float);
template string toString<double>(double);
template string toString<long double>(long double);

template<>
string toString(TemplateArg arg) {
    return arg._tparam;
}

template<>
string toString(bool val) {
    return string(val ? "true" : "false");
}

template<>
string toString(const char* str) {
    return string(str);
}

template<>
string toString(const string str) {
    return str;
}

template<>
string toString(fly_op_t val) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(fly_add_t);
        CASE_STMT(fly_sub_t);
        CASE_STMT(fly_mul_t);
        CASE_STMT(fly_div_t);

        CASE_STMT(fly_and_t);
        CASE_STMT(fly_or_t);
        CASE_STMT(fly_eq_t);
        CASE_STMT(fly_neq_t);
        CASE_STMT(fly_lt_t);
        CASE_STMT(fly_le_t);
        CASE_STMT(fly_gt_t);
        CASE_STMT(fly_ge_t);

        CASE_STMT(fly_bitnot_t);
        CASE_STMT(fly_bitor_t);
        CASE_STMT(fly_bitand_t);
        CASE_STMT(fly_bitxor_t);
        CASE_STMT(fly_bitshiftl_t);
        CASE_STMT(fly_bitshiftr_t);

        CASE_STMT(fly_min_t);
        CASE_STMT(fly_max_t);
        CASE_STMT(fly_cplx2_t);
        CASE_STMT(fly_atan2_t);
        CASE_STMT(fly_pow_t);
        CASE_STMT(fly_hypot_t);

        CASE_STMT(fly_sin_t);
        CASE_STMT(fly_cos_t);
        CASE_STMT(fly_tan_t);
        CASE_STMT(fly_asin_t);
        CASE_STMT(fly_acos_t);
        CASE_STMT(fly_atan_t);

        CASE_STMT(fly_sinh_t);
        CASE_STMT(fly_cosh_t);
        CASE_STMT(fly_tanh_t);
        CASE_STMT(fly_asinh_t);
        CASE_STMT(fly_acosh_t);
        CASE_STMT(fly_atanh_t);

        CASE_STMT(fly_exp_t);
        CASE_STMT(fly_expm1_t);
        CASE_STMT(fly_erf_t);
        CASE_STMT(fly_erfc_t);

        CASE_STMT(fly_log_t);
        CASE_STMT(fly_log10_t);
        CASE_STMT(fly_log1p_t);
        CASE_STMT(fly_log2_t);

        CASE_STMT(fly_sqrt_t);
        CASE_STMT(fly_cbrt_t);

        CASE_STMT(fly_abs_t);
        CASE_STMT(fly_cast_t);
        CASE_STMT(fly_cplx_t);
        CASE_STMT(fly_real_t);
        CASE_STMT(fly_imag_t);
        CASE_STMT(fly_conj_t);

        CASE_STMT(fly_floor_t);
        CASE_STMT(fly_ceil_t);
        CASE_STMT(fly_round_t);
        CASE_STMT(fly_trunc_t);
        CASE_STMT(fly_signbit_t);

        CASE_STMT(fly_rem_t);
        CASE_STMT(fly_mod_t);

        CASE_STMT(fly_tgamma_t);
        CASE_STMT(fly_lgamma_t);

        CASE_STMT(fly_notzero_t);

        CASE_STMT(fly_iszero_t);
        CASE_STMT(fly_isinf_t);
        CASE_STMT(fly_isnan_t);

        CASE_STMT(fly_sigmoid_t);

        CASE_STMT(fly_noop_t);

        CASE_STMT(fly_select_t);
        CASE_STMT(fly_not_select_t);
        CASE_STMT(fly_rsqrt_t);
        CASE_STMT(fly_moddims_t);

        CASE_STMT(fly_none_t);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(fly_interp_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(FLY_INTERP_NEAREST);
        CASE_STMT(FLY_INTERP_LINEAR);
        CASE_STMT(FLY_INTERP_BILINEAR);
        CASE_STMT(FLY_INTERP_CUBIC);
        CASE_STMT(FLY_INTERP_LOWER);
        CASE_STMT(FLY_INTERP_LINEAR_COSINE);
        CASE_STMT(FLY_INTERP_BILINEAR_COSINE);
        CASE_STMT(FLY_INTERP_BICUBIC);
        CASE_STMT(FLY_INTERP_CUBIC_SPLINE);
        CASE_STMT(FLY_INTERP_BICUBIC_SPLINE);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(fly_border_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(FLY_PAD_ZERO);
        CASE_STMT(FLY_PAD_SYM);
        CASE_STMT(FLY_PAD_CLAMP_TO_EDGE);
        CASE_STMT(FLY_PAD_PERIODIC);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(fly_moment_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(FLY_MOMENT_M00);
        CASE_STMT(FLY_MOMENT_M01);
        CASE_STMT(FLY_MOMENT_M10);
        CASE_STMT(FLY_MOMENT_M11);
        CASE_STMT(FLY_MOMENT_FIRST_ORDER);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(fly_match_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(FLY_SAD);
        CASE_STMT(FLY_ZSAD);
        CASE_STMT(FLY_LSAD);
        CASE_STMT(FLY_SSD);
        CASE_STMT(FLY_ZSSD);
        CASE_STMT(FLY_LSSD);
        CASE_STMT(FLY_NCC);
        CASE_STMT(FLY_ZNCC);
        CASE_STMT(FLY_SHD);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(fly_flux_function p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(FLY_FLUX_QUADRATIC);
        CASE_STMT(FLY_FLUX_EXPONENTIAL);
        CASE_STMT(FLY_FLUX_DEFAULT);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(FLY_BATCH_KIND val) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(FLY_BATCH_NONE);
        CASE_STMT(FLY_BATCH_LHS);
        CASE_STMT(FLY_BATCH_RHS);
        CASE_STMT(FLY_BATCH_SAME);
        CASE_STMT(FLY_BATCH_DIFF);
        CASE_STMT(FLY_BATCH_UNSUPPORTED);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(fly_homography_type val) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(FLY_HOMOGRAPHY_RANSAC);
        CASE_STMT(FLY_HOMOGRAPHY_LMEDS);
    }
#undef CASE_STMT
    return retVal;
}

}  // namespace common
}  // namespace flare
