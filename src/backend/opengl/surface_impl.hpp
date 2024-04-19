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
#include <shader_program.hpp>

#include <cstdint>
#include <map>

namespace theia {
namespace opengl {

class surface_impl : public AbstractRenderable {
   protected:
    /* plot points characteristics */
    uint32_t mNumXPoints;
    uint32_t mNumYPoints;
    theia::dtype mDataType;
    theia::MarkerType mMarkerType;
    /* OpenGL Objects */
    uint32_t mIBO;
    size_t mIBOSize;
    ShaderProgram mMarkerProgram;
    ShaderProgram mSurfProgram;
    /* shared variable index locations */
    uint32_t mMarkerMatIndex;
    uint32_t mMarkerPointIndex;
    uint32_t mMarkerColorIndex;
    uint32_t mMarkerAlphaIndex;
    uint32_t mMarkerPVCIndex;
    uint32_t mMarkerPVAIndex;
    uint32_t mMarkerTypeIndex;
    uint32_t mMarkerColIndex;

    uint32_t mSurfMatIndex;
    uint32_t mSurfRangeIndex;
    uint32_t mSurfPointIndex;
    uint32_t mSurfColorIndex;
    uint32_t mSurfAlphaIndex;
    uint32_t mSurfPVCIndex;
    uint32_t mSurfPVAIndex;
    uint32_t mSurfUniformColorIndex;
    uint32_t mSurfAssistDrawFlagIndex;

    std::map<int, uint32_t> mVAOMap;

    /* bind and unbind helper functions
     * for rendering resources */
    void bindResources(const int pWindowId);
    void unbindResources() const;
    glm::mat4 computeTransformMat(const glm::mat4& pView,
                                  const glm::mat4& pOrient);
    virtual void renderGraph(const int pWindowId, const glm::mat4& transform);

   public:
    surface_impl(const uint32_t pNumXpoints, const uint32_t pNumYpoints,
                 const theia::dtype pDataType,
                 const theia::MarkerType pMarkerType);
    ~surface_impl();

    void render(const int pWindowId, const int pX, const int pY, const int pVPW,
                const int pVPH, const glm::mat4& pView,
                const glm::mat4& pOrient);

    inline void usePerVertexColors(const bool pFlag = true) {
        mIsPVCOn = pFlag;
    }

    inline void usePerVertexAlphas(const bool pFlag = true) {
        mIsPVAOn = pFlag;
    }

    bool isRotatable() const { return true; }
};

class scatter3_impl : public surface_impl {
   private:
    void renderGraph(const int pWindowId, const glm::mat4& transform);

   public:
    scatter3_impl(const uint32_t pNumXPoints, const uint32_t pNumYPoints,
                  const theia::dtype pDataType,
                  const theia::MarkerType pMarkerType = FG_MARKER_NONE)
        : surface_impl(pNumXPoints, pNumYPoints, pDataType, pMarkerType) {}
};

}  // namespace opengl
}  // namespace theia
