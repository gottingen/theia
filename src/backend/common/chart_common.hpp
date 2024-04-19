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
#include <theia/glm/glm.hpp>

#include <memory>

namespace theia {
namespace common {

// Objects of type `RenderableType` in the following class definition
// should implement all the member functons of ChartRenderableBase
// class, otherwise you cannot use this class.
template<class RenderableType>
class ChartRenderableBase {
   protected:
    std::shared_ptr<RenderableType> mShrdPtr;

   public:
    ChartRenderableBase() {}

    ChartRenderableBase(const std::shared_ptr<RenderableType>& pValue)
        : mShrdPtr(pValue) {}

    inline const std::shared_ptr<RenderableType>& impl() const {
        return mShrdPtr;
    }

    inline void setColor(const float pRed, const float pGreen,
                         const float pBlue, const float pAlpha) {
        mShrdPtr->setColor(pRed, pGreen, pBlue, pAlpha);
    }

    inline void setLegend(const char* pLegend) { mShrdPtr->setLegend(pLegend); }

    inline unsigned vbo() const { return mShrdPtr->vbo(); }

    inline unsigned cbo() const { return mShrdPtr->cbo(); }

    inline unsigned abo() const { return mShrdPtr->abo(); }

    inline size_t vboSize() const { return mShrdPtr->vboSize(); }

    inline size_t cboSize() const { return mShrdPtr->cboSize(); }

    inline size_t aboSize() const { return mShrdPtr->aboSize(); }

    inline void render(const int pWindowId, const int pX, const int pY,
                       const int pVPW, const int pVPH,
                       const glm::mat4& pTransform) const {
        mShrdPtr->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
    }
};

}  // namespace common
}  // namespace theia
