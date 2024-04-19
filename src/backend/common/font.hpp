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

#include <backend.hpp>
#include <theia/fg/defines.h>
#include <font_impl.hpp>

#include <memory>

namespace theia {
namespace common {

class Font {
   private:
    std::shared_ptr<detail::font_impl> mFont;

   public:
    Font() : mFont(std::make_shared<detail::font_impl>()) {}

    Font(const fg_font pOther) {
        mFont = reinterpret_cast<Font*>(pOther)->impl();
    }

    const std::shared_ptr<detail::font_impl>& impl() const { return mFont; }

    inline void setOthro2D(int pWidth, int pHeight) {
        mFont->setOthro2D(pWidth, pHeight);
    }

    inline void loadFont(const char* const pFile) { mFont->loadFont(pFile); }

    inline void loadSystemFont(const char* const pName) {
        mFont->loadSystemFont(pName);
    }
};

}  // namespace common
}  // namespace theia
