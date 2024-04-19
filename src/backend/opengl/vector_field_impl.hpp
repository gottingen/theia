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

class vector_field_impl : public AbstractRenderable {
   protected:
    uint32_t mDimension;
    /* plot points characteristics */
    uint32_t mNumPoints;
    theia::dtype mDataType;
    /* OpenGL Objects */
    ShaderProgram mFieldProgram;
    uint32_t mDBO;
    size_t mDBOSize;
    /* shader variable index locations */
    /* vertex shader */
    uint32_t mFieldPointIndex;
    uint32_t mFieldColorIndex;
    uint32_t mFieldAlphaIndex;
    uint32_t mFieldDirectionIndex;
    /* geometry shader */
    uint32_t mFieldPVMatIndex;
    uint32_t mFieldModelMatIndex;
    uint32_t mFieldAScaleMatIndex;
    /* fragment shader */
    uint32_t mFieldPVCOnIndex;
    uint32_t mFieldPVAOnIndex;
    uint32_t mFieldUColorIndex;

    std::map<int, uint32_t> mVAOMap;

    /* bind and unbind helper functions
     * for rendering resources */
    void bindResources(const int pWindowId);
    void unbindResources() const;

    virtual glm::mat4 computeModelMatrix(const glm::mat4& pOrient);

   public:
    vector_field_impl(const uint32_t pNumPoints, const theia::dtype pDataType,
                      const int pDimension = 3);
    ~vector_field_impl();

    uint32_t directions();
    size_t directionsSize() const;

    virtual void render(const int pWindowId, const int pX, const int pY,
                        const int pVPW, const int pVPH, const glm::mat4& pView,
                        const glm::mat4& pOrient);

    virtual bool isRotatable() const;
};

class vector_field2d_impl : public vector_field_impl {
   protected:
    glm::mat4 computeModelMatrix(const glm::mat4& pOrient) override;

   public:
    vector_field2d_impl(const uint32_t pNumPoints, const theia::dtype pDataType)
        : vector_field_impl(pNumPoints, pDataType, 2) {}

    bool isRotatable() const { return false; }
};

}  // namespace opengl
}  // namespace theia
