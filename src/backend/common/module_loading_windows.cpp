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

#include <common/defines.hpp>
#include <common/module_loading.hpp>

#include <Windows.h>
#include <string>

using std::string;

namespace flare {
namespace common {

void* getFunctionPointer(LibHandle handle, const char* symbolName) {
    return GetProcAddress(handle, symbolName);
}

LibHandle loadLibrary(const char* library_name) {
    return LoadLibrary(library_name);
}

void unloadLibrary(LibHandle handle) { FreeLibrary(handle); }

string getErrorMessage() {
    const char* lpMsgBuf;
    DWORD dw = GetLastError();

    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                      FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  (LPTSTR)&lpMsgBuf, 0, NULL);
    string error_message(lpMsgBuf);
    return error_message;
}

}  // namespace common
}  // namespace flare
