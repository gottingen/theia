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
#include <string>

const char *fg_err_to_string(const fg_err err) {
    switch (err) {
        case FG_ERR_NONE: return "Success";
        case FG_ERR_SIZE: return "Invalid size";
        case FG_ERR_INVALID_TYPE: return "Invalid type";
        case FG_ERR_INVALID_ARG: return "Invalid argument";
        case FG_ERR_GL_ERROR: return "OpenGL Error";
        case FG_ERR_FREETYPE_ERROR: return "FreeType Error";
        case FG_ERR_FILE_NOT_FOUND: return "File IO Error / File Not Found";
        case FG_ERR_NOT_SUPPORTED: return "Function not supported";
        case FG_ERR_NOT_CONFIGURED: return "Function not configured to build";
        case FG_ERR_FONTCONFIG_ERROR: return "Font Config Error";
        case FG_ERR_FREEIMAGE_UNKNOWN_FORMAT:
            return "FreeImage Error: Unknown Format";
        case FG_ERR_FREEIMAGE_BAD_ALLOC: return "FreeImage Error: Bad Alloc";
        case FG_ERR_FREEIMAGE_SAVE_FAILED:
            return "FreeImage Error: Save file failed";
        case FG_ERR_INTERNAL: return "Internal Error";
        case FG_ERR_RUNTIME: return "Runtime Error";
        case FG_ERR_UNKNOWN:
        default: return "Unknown Error";
    }
}

void fg_get_last_error(char **msg, int *len) {
    std::string &error = theia::common::getGlobalErrorString();
    int slen = std::min(theia::common::MAX_ERR_SIZE, (int)error.size());
    if (len && slen == 0) {
        *len = 0;
        *msg = NULL;
        return;
    }

    *msg = new char[slen + 1];
    error.copy(*msg, slen);
    (*msg)[slen] = '\0';
    error        = "";

    if (len) *len = slen;
}
