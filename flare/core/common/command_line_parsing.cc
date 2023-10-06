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


#include <flare/core/common/command_line_parsing.h>
#include <flare/core/common/error.h>

#include <cstring>
#include <iostream>
#include <regex>
#include <string>
#include <sstream>

namespace {

    auto const regex_true = std::regex(
            "(yes|true|1)", std::regex_constants::icase | std::regex_constants::egrep);

    auto const regex_false = std::regex(
            "(no|false|0)", std::regex_constants::icase | std::regex_constants::egrep);

}  // namespace

bool flare::detail::is_unsigned_int(const char *str) {
    const size_t len = strlen(str);
    for (size_t i = 0; i < len; ++i) {
        if (!isdigit(str[i])) {
            return false;
        }
    }
    return true;
}

bool flare::detail::check_arg(char const *arg, char const *expected) {
    std::size_t arg_len = std::strlen(arg);
    std::size_t exp_len = std::strlen(expected);
    if (arg_len < exp_len) return false;
    if (std::strncmp(arg, expected, exp_len) != 0) return false;
    if (arg_len == exp_len) return true;

    if (std::isalnum(arg[exp_len]) || arg[exp_len] == '-' ||
        arg[exp_len] == '_') {
        return false;
    }
    return true;
}

bool flare::detail::check_env_bool(char const *name, bool &val) {
    char const *var = std::getenv(name);

    if (!var) {
        return false;
    }

    if (std::regex_match(var, regex_true)) {
        val = true;
        return true;
    }

    if (!std::regex_match(var, regex_false)) {
        std::stringstream ss;
        ss << "Error: cannot convert environment variable '" << name << "=" << var
           << "' to a boolean."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }

    val = false;
    return true;
}

bool flare::detail::check_env_int(char const *name, int &val) {
    char const *var = std::getenv(name);

    if (!var) {
        return false;
    }

    errno = 0;
    char *var_end;
    val = std::strtol(var, &var_end, 10);

    if (var == var_end) {
        std::stringstream ss;
        ss << "Error: cannot convert environment variable '" << name << '=' << var
           << "' to an integer."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }

    if (errno == ERANGE) {
        std::stringstream ss;
        ss << "Error: converted value for environment variable '" << name << '='
           << var << "' falls out of range."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }

    return true;
}

bool flare::detail::check_arg_bool(char const *arg, char const *name,
                                   bool &val) {
    auto const len = std::strlen(name);
    if (std::strncmp(arg, name, len) != 0) {
        return false;
    }
    auto const arg_len = strlen(arg);
    if (arg_len == len) {
        val = true;  // --flare-foo without =BOOL interpreted as fool=true
        return true;
    }
    if (arg_len <= len + 1 || arg[len] != '=') {
        std::stringstream ss;
        ss << "Error: command line argument '" << arg
           << "' is not recognized as a valid boolean."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }

    std::advance(arg, len + 1);
    if (std::regex_match(arg, regex_true)) {
        val = true;
        return true;
    }
    if (!std::regex_match(arg, regex_false)) {
        std::stringstream ss;
        ss << "Error: cannot convert command line argument '" << name << "=" << arg
           << "' to a boolean."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }
    val = false;
    return true;
}

bool flare::detail::check_arg_int(char const *arg, char const *name, int &val) {
    auto const len = std::strlen(name);
    if (std::strncmp(arg, name, len) != 0) {
        return false;
    }
    auto const arg_len = strlen(arg);
    if (arg_len <= len + 1 || arg[len] != '=') {
        std::stringstream ss;
        ss << "Error: command line argument '" << arg
           << "' is not recognized as a valid integer."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }

    std::advance(arg, len + 1);

    errno = 0;
    char *arg_end;
    val = std::strtol(arg, &arg_end, 10);

    if (arg == arg_end) {
        std::stringstream ss;
        ss << "Error: cannot convert command line argument '" << name << '=' << arg
           << "' to an integer."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }

    if (errno == ERANGE) {
        std::stringstream ss;
        ss << "Error: converted value for command line argument '" << name << '='
           << arg << "' falls out of range."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }

    return true;
}

bool flare::detail::check_arg_str(char const *arg, char const *name,
                                  std::string &val) {
    auto const len = std::strlen(name);
    if (std::strncmp(arg, name, len) != 0) {
        return false;
    }
    auto const arg_len = strlen(arg);
    if (arg_len <= len + 1 || arg[len] != '=') {
        std::stringstream ss;
        ss << "Error: command line argument '" << arg
           << "' is not recognized as a valid string."
           << " Raised by flare::initialize().\n";
        flare::abort(ss.str().c_str());
    }

    std::advance(arg, len + 1);

    val = arg;
    return true;
}

void flare::detail::warn_deprecated_environment_variable(
        std::string deprecated) {
    std::cerr << "Warning: environment variable '" << deprecated
              << "' is deprecated."
              << " Raised by flare::initialize()." << std::endl;
}

void flare::detail::warn_deprecated_environment_variable(
        std::string deprecated, std::string use_instead) {
    std::cerr << "Warning: environment variable '" << deprecated
              << "' is deprecated."
              << " Use '" << use_instead << "' instead."
              << " Raised by flare::initialize()." << std::endl;
}

void flare::detail::warn_deprecated_command_line_argument(
        std::string deprecated) {
    std::cerr << "Warning: command line argument '" << deprecated
              << "' is deprecated."
              << " Raised by flare::initialize()." << std::endl;
}

void flare::detail::warn_deprecated_command_line_argument(
        std::string deprecated, std::string use_instead) {
    std::cerr << "Warning: command line argument '" << deprecated
              << "' is deprecated."
              << " Use '" << use_instead << "' instead."
              << " Raised by flare::initialize()." << std::endl;
}

namespace {
    std::vector<std::regex> do_not_warn_regular_expressions{
            std::regex{"--flare-tool.*", std::regex::egrep},
    };
}

void flare::detail::do_not_warn_not_recognized_command_line_argument(
        std::regex ignore) {
    do_not_warn_regular_expressions.push_back(std::move(ignore));
}

void flare::detail::warn_not_recognized_command_line_argument(
        std::string not_recognized) {
    for (auto const &ignore: do_not_warn_regular_expressions) {
        if (std::regex_match(not_recognized, ignore)) {
            return;
        }
    }
    std::cerr << "Warning: command line argument '" << not_recognized
              << "' is not recognized."
              << " Raised by flare::initialize()." << std::endl;
}
