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

/**
 *
 * A font atlas is used to pack several small regions into a single texture.
 *
 * It is an implementation of Skyline Bottom-Left algorithm described
 * in the article by Jukka Jylänki : "A  Thousand Ways to Pack the Bin -
 * A Practical Approach to Two-Dimensional Rectangle Bin Packing",
 * February 27, 2010. Following code also loosely follows C++ sources provided
 * by Jukka Jylänki at: http://clb.demon.fi/files/RectangleBinPack/ for the
 * implementation of the Skyline Bottom-Left algorithm.
 */

#pragma once

#include <common/defines.hpp>

namespace theia {
namespace opengl {

class FontAtlas {
   private:
    size_t mWidth;
    size_t mHeight;
    size_t mDepth;
    size_t mUsed;
    uint32_t mId;

    std::vector<unsigned char> mData;
    std::vector<glm::vec3> nodes;

    /* helper functions */
    int fit(const size_t pIndex, const size_t pWidth, const size_t pHeight);
    void merge();

   public:
    FontAtlas(const size_t pWidth, const size_t pHeight, const size_t pDepth);
    ~FontAtlas();

    size_t width() const;
    size_t height() const;
    size_t depth() const;

    glm::vec4 getRegion(const size_t pWidth, const size_t pHeight);
    bool setRegion(const size_t pX, const size_t pY, const size_t pWidth,
                   const size_t pHeight, const unsigned char* pData,
                   const size_t pStride);

    void upload();
    void clear();

    uint32_t atlasTextureId() const;
};

struct Glyph {
    size_t mWidth;
    size_t mHeight;

    int mBearingX;
    int mBearingY;

    float mAdvanceX;
    float mAdvanceY;

    /* normalized texture coordinate (x) of top-left corner */
    float mS0, mT0;

    /* First normalized texture coordinate (x) of bottom-right corner */
    float mS1, mT1;

    /* render quad vbo offset */
    size_t mOffset;
};

}  // namespace opengl
}  // namespace theia
