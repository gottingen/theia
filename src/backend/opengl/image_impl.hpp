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

#include <abstract_renderable.hpp>
#include <theia/fg/defines.h>
#include <shader_program.hpp>

#include <cstdint>

namespace theia {
namespace opengl {

class image_impl : public AbstractRenderable {
   private:
    uint32_t mWidth;
    uint32_t mHeight;
    theia::ChannelFormat mFormat;
    theia::dtype mDataType;
    float mAlpha;
    bool mKeepARatio;
    size_t mFormatSize;
    /* internal resources for interop */
    size_t mPBOsize;
    uint32_t mPBO;
    uint32_t mTex;
    ShaderProgram mProgram;
    uint32_t mMatIndex;
    uint32_t mTexIndex;
    uint32_t mNumCIndex;
    uint32_t mAlphaIndex;
    uint32_t mCMapLenIndex;
    uint32_t mCMapIndex;
    /* color map details */
    uint32_t mColorMapUBO;
    uint32_t mUBOSize;

    /* helper functions to bind and unbind
     * resources for render quad primitive */
    void bindResources(int pWindowId) const;
    void unbindResources() const;

   public:
    image_impl(const uint32_t pWidth, const uint32_t pHeight,
               const theia::ChannelFormat pFormat,
               const theia::dtype pDataType);
    ~image_impl();

    void setColorMapUBOParams(const uint32_t pUBO, const uint32_t pSize);
    void setAlpha(const float pAlpha);
    void keepAspectRatio(const bool pKeep = true);

    uint32_t width() const;
    uint32_t height() const;
    theia::ChannelFormat pixelFormat() const;
    theia::dtype channelType() const;
    uint32_t pbo() const;
    uint32_t size() const;

    void render(const int pWindowId, const int pX, const int pY, const int pVPW,
                const int pVPH, const glm::mat4 &pView,
                const glm::mat4 &pOrient);

    bool isRotatable() const;
};

}  // namespace opengl
}  // namespace theia
