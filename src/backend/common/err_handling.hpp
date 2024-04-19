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

#pragma once

#include <common/defines.hpp>
#include <theia/fg/exception.h>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace theia {
namespace common {
////////////////////////////////////////////////////////////////////////////////
// Exception Classes
// Error, TypeError, ArgumentError
////////////////////////////////////////////////////////////////////////////////
class FgError : public std::logic_error {
    std::string mFuncName;
    std::string mFileName;
    int mLineNumber;
    theia::ErrorCode mErrCode;
    FgError();

   public:
    FgError(const char* const pFuncName, const char* const pFileName,
            const int pLineNumber, const char* const pMessage,
            theia::ErrorCode pErrCode);

    FgError(std::string pFuncName, std::string pFileName, const int pLineNumber,
            std::string pMessage, theia::ErrorCode pErrCode);

    const std::string& getFunctionName() const { return mFuncName; }

    const std::string& getFileName() const { return mFileName; }

    int getLineNumber() const { return mLineNumber; }

    theia::ErrorCode getError() const { return mErrCode; }

    virtual ~FgError() throw();
};

// TODO: Perhaps add a way to return supported types
class TypeError : public FgError {
    int mArgIndex;
    std::string mErrTypeName;

    TypeError();

   public:
    TypeError(const char* const pFuncName, const char* const pFileName,
              const int pLine, const int pIndex, const theia::dtype pType);

    const std::string& getTypeName() const { return mErrTypeName; }

    int getArgIndex() const { return mArgIndex; }

    ~TypeError() throw() {}
};

class ArgumentError : public FgError {
    int mArgIndex;
    std::string mExpected;

    ArgumentError();

   public:
    ArgumentError(const char* const pFuncName, const char* const pFileName,
                  const int pLine, const int pIndex,
                  const char* const pExpectString);

    const std::string& getExpectedCondition() const { return mExpected; }

    int getArgIndex() const { return mArgIndex; }

    ~ArgumentError() throw() {}
};

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////
static const int MAX_ERR_SIZE = 1024;

std::string& getGlobalErrorString();

void print_error(const std::string& msg);

fg_err processException();

const char* getName(theia::dtype type);

////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////
#define ARG_ASSERT(INDEX, COND)                                                \
    do {                                                                       \
        if ((COND) == false) {                                                 \
            throw theia::common::ArgumentError(                                \
                __PRETTY_FUNCTION__, __FG_FILENAME__, __LINE__, INDEX, #COND); \
        }                                                                      \
    } while (0)

#define TYPE_ERROR(INDEX, type)                                              \
    do {                                                                     \
        throw theia::common::TypeError(__PRETTY_FUNCTION__, __FG_FILENAME__, \
                                       __LINE__, INDEX, type);               \
    } while (0)

#define FG_ERROR(MSG, ERR_TYPE)                                            \
    do {                                                                   \
        throw theia::common::FgError(__PRETTY_FUNCTION__, __FG_FILENAME__, \
                                     __LINE__, MSG, ERR_TYPE);             \
    } while (0)

#define TYPE_ASSERT(COND)                                       \
    do {                                                        \
        if ((COND) == false) {                                  \
            FG_ERROR("Type mismatch inputs", FG_ERR_DIFF_TYPE); \
        }                                                       \
    } while (0)

#define CATCHALL                                  \
    catch (...) {                                 \
        return theia::common::processException(); \
    }

}  // namespace common
}  // namespace theia
