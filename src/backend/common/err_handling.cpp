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
#include <common/util.hpp>
#include <theia/fg/exception.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

namespace theia {
namespace common {

using std::string;
using std::stringstream;

FgError::FgError(const char *const pFuncName, const char *const pFileName,
                 const int pLineNumber, const char *const pMessage,
                 ErrorCode pErrCode)
    : logic_error(pMessage)
    , mFuncName(pFuncName)
    , mFileName(pFileName)
    , mLineNumber(pLineNumber)
    , mErrCode(pErrCode) {}

FgError::FgError(string pFuncName, string pFileName, const int pLineNumber,
                 string pMessage, ErrorCode pErrCode)
    : logic_error(pMessage)
    , mFuncName(pFuncName)
    , mFileName(pFileName)
    , mLineNumber(pLineNumber)
    , mErrCode(pErrCode) {}

FgError::~FgError() throw() {}

TypeError::TypeError(const char *const pFuncName, const char *const pFileName,
                     const int pLineNumber, const int pIndex,
                     const theia::dtype pType)
    : FgError(pFuncName, pFileName, pLineNumber, "Invalid data type",
              FG_ERR_INVALID_TYPE)
    , mArgIndex(pIndex)
    , mErrTypeName(getName(pType)) {}

ArgumentError::ArgumentError(const char *const pFuncName,
                             const char *const pFileName, const int pLineNumber,
                             const int pIndex, const char *const pExpectString)
    : FgError(pFuncName, pFileName, pLineNumber, "Invalid argument",
              FG_ERR_INVALID_ARG)
    , mArgIndex(pIndex)
    , mExpected(pExpectString) {}

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////
std::string &getGlobalErrorString() {
    static std::string global_error_string = std::string("");
    return global_error_string;
}

void print_error(const string &msg) {
    std::string perr = getEnvVar("FG_PRINT_ERRORS");
    if (!perr.empty()) {
        if (perr != "0") fprintf(stderr, "%s\n", msg.c_str());
    }

    getGlobalErrorString() = msg;
}

fg_err processException() {
    stringstream ss;
    fg_err err = FG_ERR_INTERNAL;

    try {
        throw;
    } catch (const TypeError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLineNumber()
           << "\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n"
           << "Expects the type : " << ex.getTypeName() << "\n";

        print_error(ss.str());
        err = FG_ERR_INVALID_TYPE;
    } catch (const ArgumentError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLineNumber()
           << "\n"
           << "Invalid argument at index " << ex.getArgIndex() << "\n"
           << "Expected : " << ex.getExpectedCondition() << "\n";

        print_error(ss.str());
        err = FG_ERR_INVALID_ARG;
    } catch (const FgError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLineNumber()
           << "\n"
           << ex.what() << "\n";

        print_error(ss.str());
        err = ex.getError();
    } catch (...) {
        print_error(ss.str());
        err = FG_ERR_UNKNOWN;
    }

    return err;
}

const char *getName(theia::dtype type) {
    switch (type) {
        case s8: return "char";
        case u8: return "unsigned char";
        case s32: return "int";
        case u32: return "unsigned int";
        case f32: return "float";
        case s16: return "short";
        case u16: return "unsigned short";
        default: TYPE_ERROR(1, type);
    }
}

}  // namespace common
}  // namespace theia
