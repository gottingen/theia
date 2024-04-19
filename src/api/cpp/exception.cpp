// Copyright 2024 The Turbo Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <common/err_handling.hpp>
#include <theia/fg/exception.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

using std::cerr;
using std::string;
using std::stringstream;

namespace theia {

void stringcopy(char* dest, const char* src, size_t len) {
#if defined(OS_WIN)
    strncpy_s(dest, theia::common::MAX_ERR_SIZE, src, len);
#else
    std::strncpy(dest, src, len);
#endif
}

Error::Error() : mErrCode(FG_ERR_UNKNOWN) {
    stringcopy(mMessage, "Unknown Exception", sizeof(mMessage));
}

Error::Error(const char* const pMessage) : mErrCode(FG_ERR_UNKNOWN) {
    stringcopy(mMessage, pMessage, sizeof(mMessage));
    mMessage[sizeof(mMessage) - 1] = '\0';
}

Error::Error(const char* const pFileName, int pLine, ErrorCode pErrCode)
    : mErrCode(pErrCode) {
    std::snprintf(mMessage, sizeof(mMessage) - 1,
                  "theia Exception (%s:%d):\nIn %s:%d",
                  fg_err_to_string(pErrCode), (int)pErrCode, pFileName, pLine);
    mMessage[sizeof(mMessage) - 1] = '\0';
}

Error::Error(const char* const pMessage, const char* const pFileName,
             const int pLine, ErrorCode pErrCode)
    : mErrCode(pErrCode) {
    std::snprintf(mMessage, sizeof(mMessage) - 1,
                  "theia Exception (%s:%d):\n%s\nIn %s:%d",
                  fg_err_to_string(pErrCode), (int)pErrCode, pMessage,
                  pFileName, pLine);
    mMessage[sizeof(mMessage) - 1] = '\0';
}

Error::Error(const char* const pMessage, const char* const pFuncName,
             const char* const pFileName, const int pLine, ErrorCode pErrCode)
    : mErrCode(pErrCode) {
    std::snprintf(mMessage, sizeof(mMessage) - 1,
                  "theia Exception (%s:%d):\n%sIn function %s\nIn file %s:%d",
                  fg_err_to_string(pErrCode), (int)pErrCode, pMessage,
                  pFuncName, pFileName, pLine);
    mMessage[sizeof(mMessage) - 1] = '\0';
}

Error::Error(const Error& error) {
    this->mErrCode = error.err();
    memcpy(this->mMessage, error.what(), 1024);
}

Error::~Error() throw() {}

}  // namespace theia
