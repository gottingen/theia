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
#include <map>

namespace theia {
namespace opengl {

class histogram_impl : public AbstractRenderable {
   private:
    /* plot points characteristics */
    theia::dtype mDataType;
    uint32_t mNBins;
    /* OpenGL Objects */
    ShaderProgram mProgram;
    /* internal shader attributes for mProgram
     * shader program to render histogram bars for each
     * bin*/
    uint32_t mYMaxIndex;
    uint32_t mNBinsIndex;
    uint32_t mMatIndex;
    uint32_t mPointIndex;
    uint32_t mFreqIndex;
    uint32_t mColorIndex;
    uint32_t mAlphaIndex;
    uint32_t mPVCIndex;
    uint32_t mPVAIndex;
    uint32_t mBColorIndex;

    std::map<int, uint32_t> mVAOMap;

    /* bind and unbind helper functions
     * for rendering resources */
    void bindResources(const int pWindowId);
    void unbindResources() const;

   public:
    histogram_impl(const uint32_t pNBins, const theia::dtype pDataType);
    ~histogram_impl();

    void render(const int pWindowId, const int pX, const int pY, const int pVPW,
                const int pVPH, const glm::mat4 &pView,
                const glm::mat4 &pOrient);

    bool isRotatable() const;
};

}  // namespace opengl
}  // namespace theia
