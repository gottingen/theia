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

#include <theia/fg/font.h>

#include <error.hpp>

#include <utility>

namespace theia {
Font::Font() {
    fg_font temp = 0;
    FG_THROW(fg_create_font(&temp));
    std::swap(mValue, temp);
}

Font::Font(const Font& other) {
    fg_font temp = 0;
    FG_THROW(fg_retain_font(&temp, other.get()));
    std::swap(mValue, temp);
}

Font::~Font() { fg_release_font(get()); }

void Font::loadFontFile(const char* const pFile) {
    FG_THROW(fg_load_font_file(get(), pFile));
}

void Font::loadSystemFont(const char* const pName) {
    FG_THROW(fg_load_system_font(get(), pName));
}

fg_font Font::get() const { return mValue; }
}  // namespace theia
