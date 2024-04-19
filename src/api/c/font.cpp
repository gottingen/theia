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

#include <common/font.hpp>
#include <common/handle.hpp>
#include <theia/fg/font.h>

using namespace theia;

using theia::common::getFont;

fg_err fg_create_font(fg_font* pFont) {
    try {
        *pFont = getHandle(new common::Font());
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_font(fg_font* pOut, fg_font pIn) {
    try {
        common::Font* temp = new common::Font(getFont(pIn));
        *pOut              = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_font(fg_font pFont) {
    try {
        delete getFont(pFont);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_load_font_file(fg_font pFont, const char* const pFileFullPath) {
    try {
        getFont(pFont)->loadFont(pFileFullPath);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_load_system_font(fg_font pFont, const char* const pFontName) {
    try {
        getFont(pFont)->loadSystemFont(pFontName);
    }
    CATCHALL

    return FG_ERR_NONE;
}
