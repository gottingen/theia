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

#include <theia/fg/defines.h>

#ifdef __cplusplus
#include <iostream>
#include <stdexcept>

namespace theia
{

/// \brief Error is exception object thrown by theia for internal errors
class FGAPI Error : public std::exception
{
private:

    char        mMessage[1024];

    ErrorCode   mErrCode;

public:

    ErrorCode err() const { return mErrCode; }

    Error();

    Error(const char * const pMessage);

    Error(const char * const pFileName, int pLine, ErrorCode pErrCode);

    Error(const char * const pMessage, const char * const pFileName, int pLine, ErrorCode pErrCode);

    Error(const char * const pMessage, const char * const pFuncName,
          const char * const pFileName, int pLine, ErrorCode pErrCode);

    Error(const Error& error);

    virtual ~Error() throw();

    virtual const char * what() const throw() { return mMessage; }

    friend inline std::ostream& operator<<(std::ostream &s, const Error &e)
    { return s << e.what(); }
};

} // namespace theia

#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
    Fetch the last error's error code

    \ingroup util_functions
 */
FGAPI void fg_get_last_error(char **msg, int *len);

/**
    Fetch the string message associated to given error code

    \ingroup util_functions
 */
FGAPI const char * fg_err_to_string(const fg_err err);

#ifdef __cplusplus
}
#endif
