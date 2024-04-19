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
#include <chart_impl.hpp>
#include <common/err_handling.hpp>
#include <theia/fg/defines.h>
#include <theia/glm/glm.hpp>

#include <memory>

namespace theia {
namespace common {

class Chart {
   private:
    theia::ChartType mChartType;
    std::shared_ptr<detail::AbstractChart> mChart;

   public:
    Chart(const theia::ChartType cType) : mChartType(cType) {
        ARG_ASSERT(0, cType == FG_CHART_2D || cType == FG_CHART_3D);

        if (cType == FG_CHART_2D) {
            mChart = std::make_shared<detail::chart2d_impl>();
        } else if (cType == FG_CHART_3D) {
            mChart = std::make_shared<detail::chart3d_impl>();
        }
    }

    Chart(const fg_chart pOther) {
        mChart = reinterpret_cast<Chart*>(pOther)->impl();
    }

    inline theia::ChartType chartType() const { return mChartType; }

    inline const std::shared_ptr<detail::AbstractChart>& impl() const {
        return mChart;
    }

    void setAxesVisibility(const bool isVisible = true) {
        mChart->setAxesVisibility(isVisible);
    }

    inline void setAxesTitles(const char* pX, const char* pY, const char* pZ) {
        mChart->setAxesTitles(pX, pY, pZ);
    }

    inline void setAxesLimits(const float pXmin, const float pXmax,
                              const float pYmin, const float pYmax,
                              const float pZmin, const float pZmax) {
        mChart->setAxesLimits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax);
    }

    inline void setAxesLabelFormat(const std::string& pXFormat,
                                   const std::string& pYFormat,
                                   const std::string& pZFormat) {
        mChart->setAxesLabelFormat(pXFormat, pYFormat, pZFormat);
    }

    inline void getAxesLimits(float* pXmin, float* pXmax, float* pYmin,
                              float* pYmax, float* pZmin, float* pZmax) {
        mChart->getAxesLimits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax);
    }

    inline void setLegendPosition(const float pX, const float pY) {
        mChart->setLegendPosition(pX, pY);
    }

    inline void addRenderable(
        const std::shared_ptr<detail::AbstractRenderable> pRenderable) {
        mChart->addRenderable(pRenderable);
    }

    inline void removeRenderable(
        const std::shared_ptr<detail::AbstractRenderable> pRenderable) {
        mChart->removeRenderable(pRenderable);
    }

    inline void render(const int pWindowId, const int pX, const int pY,
                       const int pVPW, const int pVPH, const glm::mat4& pView,
                       const glm::mat4& pOrient) const {
        mChart->render(pWindowId, pX, pY, pVPW, pVPH, pView, pOrient);
    }
};

}  // namespace common
}  // namespace theia
