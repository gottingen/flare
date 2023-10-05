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

#ifndef FLARE_CORE_COMMON_COMMAND_LINE_PARSING_H_
#define FLARE_CORE_COMMON_COMMAND_LINE_PARSING_H_

#include <string>
#include <regex>

namespace flare::detail {
    bool is_unsigned_int(const char *str);

    bool check_arg(char const *arg, char const *expected);

    bool check_arg_bool(char const *arg, char const *name, bool &val);

    bool check_arg_int(char const *arg, char const *name, int &val);

    bool check_arg_str(char const *arg, char const *name, std::string &val);

    bool check_env_bool(char const *name, bool &val);

    bool check_env_int(char const *name, int &val);

    void warn_deprecated_environment_variable(std::string deprecated);

    void warn_deprecated_environment_variable(std::string deprecated,
                                              std::string use_instead);

    void warn_deprecated_command_line_argument(std::string deprecated);

    void warn_deprecated_command_line_argument(std::string deprecated,
                                               std::string use_instead);

    void warn_not_recognized_command_line_argument(std::string not_recognized);

    void do_not_warn_not_recognized_command_line_argument(std::regex ignore);

}  // namespace flare::detail

#endif  // FLARE_CORE_COMMON_COMMAND_LINE_PARSING_H_
