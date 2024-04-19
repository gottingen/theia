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

#include <font_atlas_impl.hpp>
#include <shader_program.hpp>

#include <map>

namespace theia {
namespace opengl {

typedef std::vector<Glyph*> GlyphList;

class font_impl {
   private:
    /* VAO map to store a vertex array object
     * for each valid window context */
    std::map<int, unsigned int> mVAOMap;

    /* attributes */
    std::string mTTFfile;
    bool mIsFontLoaded;
    std::unique_ptr<FontAtlas> mAtlas;
    unsigned int mVBO;
    ShaderProgram mProgram;
    size_t mOrthoW;
    size_t mOrthoH;

    std::vector<GlyphList> mGlyphLists;

    /* OpenGL Data */
    glm::mat4 mProjMat;
    unsigned int mPMatIndex;
    unsigned int mMMatIndex;
    unsigned int mTexIndex;
    unsigned int mClrIndex;

    /* load all glyphs and create character atlas */
    void loadAtlasWithGlyphs(const size_t pFontSize);

    /* helper functions to bind and unbind
     * rendering resources */
    void bindResources(int pWindowId);
    void unbindResources() const;

    /* helper to destroy GL objects created for
     * given font face and size if required */
    void destroyGLResources();

   public:
    font_impl();
    ~font_impl();

    void setOthro2D(size_t pWidth, size_t pHeight);
    void loadFont(const char* const pFile);
    void loadSystemFont(const char* const pName);

    void render(int pWindowId, const float pPos[2], const float pColor[4],
                const char* pText, size_t pFontSize, bool pIsVertical = false);
};

}  // namespace opengl
}  // namespace theia
