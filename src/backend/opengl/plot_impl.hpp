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

class plot_impl : public AbstractRenderable {
   protected:
    uint32_t mDimension;
    float mMarkerSize;
    /* plot points characteristics */
    uint32_t mNumPoints;
    theia::dtype mDataType;
    theia::MarkerType mMarkerType;
    theia::PlotType mPlotType;
    bool mIsPVROn;
    /* OpenGL Objects */
    ShaderProgram mPlotProgram;
    ShaderProgram mMarkerProgram;
    uint32_t mRBO;
    size_t mRBOSize;
    /* shader variable index locations */
    uint32_t mPlotMatIndex;
    uint32_t mPlotPVCOnIndex;
    uint32_t mPlotPVAOnIndex;
    uint32_t mPlotUColorIndex;
    uint32_t mPlotRangeIndex;
    uint32_t mPlotPointIndex;
    uint32_t mPlotColorIndex;
    uint32_t mPlotAlphaIndex;
    uint32_t mPlotAssistDrawFlagIndex;
    uint32_t mPlotLineColorIndex;

    uint32_t mMarkerPVCOnIndex;
    uint32_t mMarkerPVAOnIndex;
    uint32_t mMarkerPVROnIndex;
    uint32_t mMarkerTypeIndex;
    uint32_t mMarkerColIndex;
    uint32_t mMarkerMatIndex;
    uint32_t mMarkerPSizeIndex;
    uint32_t mMarkerPointIndex;
    uint32_t mMarkerColorIndex;
    uint32_t mMarkerAlphaIndex;
    uint32_t mMarkerRadiiIndex;

    std::map<int, uint32_t> mVAOMap;

    /* bind and unbind helper functions
     * for rendering resources */
    void bindResources(const int pWindowId);
    void unbindResources() const;

    virtual glm::mat4 computeTransformMat(const glm::mat4& pView,
                                          const glm::mat4& pOrient);

    virtual void
    bindDimSpecificUniforms();  // has to be called only after shaders are bound

   public:
    plot_impl(const uint32_t pNumPoints, const theia::dtype pDataType,
              const theia::PlotType pPlotType,
              const theia::MarkerType pMarkerType, const int pDimension = 3,
              const bool pIsInternalObject = false);
    ~plot_impl();

    void setMarkerSize(const float pMarkerSize);

    uint32_t markers();
    size_t markersSizes() const;

    virtual void render(const int pWindowId, const int pX, const int pY,
                        const int pVPW, const int pVPH, const glm::mat4& pView,
                        const glm::mat4& pOrient);

    virtual bool isRotatable() const;
};

class plot2d_impl : public plot_impl {
   protected:
    glm::mat4 computeTransformMat(const glm::mat4& pView,
                                  const glm::mat4& pOrient) override;

    void bindDimSpecificUniforms()
        override;  // has to be called only after shaders are bound

   public:
    plot2d_impl(const uint32_t pNumPoints, const theia::dtype pDataType,
                const theia::PlotType pPlotType,
                const theia::MarkerType pMarkerType)
        : plot_impl(pNumPoints, pDataType, pPlotType, pMarkerType, 2) {}

    bool isRotatable() const;
};

}  // namespace opengl
}  // namespace theia
