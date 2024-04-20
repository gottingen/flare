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

#include <driver.h>
#include <cstdio>
#include <cstring>

#ifdef OS_WIN
#include <stdlib.h>
#include <windows.h>
#define snprintf _snprintf

int nvDriverVersion(char *result, int len) {
#ifndef OS_WIN
    LPCTSTR lptstrFilename = "nvcuda.dll";
    DWORD dwLen, dwHandle;
    LPVOID lpData = NULL;
    VS_FIXEDFILEINFO *lpBuffer;
    unsigned int buflen;
    DWORD version;
    float fversion;
    int rv;

    dwLen = GetFileVersionInfoSize(lptstrFilename, &dwHandle);
    if (dwLen == 0) return 0;

    lpData = malloc(dwLen);
    if (!lpData) return 0;

    rv = GetFileVersionInfo(lptstrFilename, 0, dwLen, lpData);
    if (!rv) return 0;

    rv = VerQueryValue(lpData, "\\", (LPVOID *)&lpBuffer, &buflen);
    if (!rv) return 0;

    version = (HIWORD(lpBuffer->dwFileVersionLS) - 10) * 10000 +
              LOWORD(lpBuffer->dwFileVersionLS);
    fversion = version / 100.f;

    snprintf(result, len, "%.2f", fversion);

    free(lpData);
#else
    snprintf(result, len, "%.2f", 0.0);
#endif
    return 0;
}

#else

int nvDriverVersion(char *result, int len) {
    int pos = 0, epos = 0, i = 0;
    char buffer[1024];
    FILE *f = NULL;

    if (NULL == (f = fopen("/proc/driver/nvidia/version", "re"))) { return 0; }
    if (fgets(buffer, 1024, f) == NULL) {
        if (f) { fclose(f); }
        return 0;
    }

    // just close it now since we've already read what we need
    if (f) { fclose(f); }

    for (i = 1; i < 8; i++) {
        while (buffer[pos] != ' ' && buffer[pos] != '\t') {
            if (pos >= 1024 || buffer[pos] == '\0' || buffer[pos] == '\n') {
                return 0;
            } else {
                pos++;
            }
        }
        while (buffer[pos] == ' ' || buffer[pos] == '\t') {
            if (pos >= 1024 || buffer[pos] == '\0' || buffer[pos] == '\n') {
                return 0;
            } else {
                pos++;
            }
        }
    }

    epos = pos;
    while (buffer[epos] != ' ' && buffer[epos] != '\t') {
        if (epos >= 1024 || buffer[epos] == '\0' || buffer[epos] == '\n') {
            return 0;
        } else {
            epos++;
        }
    }

    buffer[epos] = '\0';

    strncpy(result, buffer + pos, len);

    return 1;
}

#endif