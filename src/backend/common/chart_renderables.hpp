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
#include <common/chart_common.hpp>
#include <histogram_impl.hpp>
#include <plot_impl.hpp>
#include <surface_impl.hpp>
#include <vector_field_impl.hpp>

#include <memory>

namespace theia {
namespace common {

class Histogram : public ChartRenderableBase<detail::histogram_impl> {
   public:
    Histogram(unsigned pNBins, theia::dtype pDataType)
        : ChartRenderableBase<detail::histogram_impl>(
              std::make_shared<detail::histogram_impl>(pNBins, pDataType)) {}

    Histogram(const fg_histogram pOther)
        : ChartRenderableBase<detail::histogram_impl>(
              reinterpret_cast<Histogram *>(pOther)->impl()) {}
};

class Plot : public ChartRenderableBase<detail::plot_impl> {
   public:
    Plot(const unsigned pNumPoints, const theia::dtype pDataType,
         const theia::PlotType pPlotType, const theia::MarkerType pMarkerType,
         const theia::ChartType pChartType) {
        if (pChartType == FG_CHART_2D) {
            mShrdPtr = std::make_shared<detail::plot2d_impl>(
                pNumPoints, pDataType, pPlotType, pMarkerType);
        } else {
            mShrdPtr = std::make_shared<detail::plot_impl>(
                pNumPoints, pDataType, pPlotType, pMarkerType);
        }
    }

    Plot(const fg_plot pOther)
        : ChartRenderableBase<detail::plot_impl>(
              reinterpret_cast<Plot *>(pOther)->impl()) {}

    inline void setMarkerSize(const float pMarkerSize) {
        mShrdPtr->setMarkerSize(pMarkerSize);
    }

    inline unsigned mbo() const { return mShrdPtr->markers(); }

    inline size_t mboSize() const { return mShrdPtr->markersSizes(); }
};

class Surface : public ChartRenderableBase<detail::surface_impl> {
   public:
    Surface(const unsigned pNumXPoints, const unsigned pNumYPoints,
            const theia::dtype pDataType,
            const theia::PlotType pPlotType     = FG_PLOT_SURFACE,
            const theia::MarkerType pMarkerType = FG_MARKER_NONE) {
        switch (pPlotType) {
            case (FG_PLOT_SURFACE):
                mShrdPtr = std::make_shared<detail::surface_impl>(
                    pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                break;
            case (FG_PLOT_SCATTER):
                mShrdPtr = std::make_shared<detail::scatter3_impl>(
                    pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                break;
            default:
                mShrdPtr = std::make_shared<detail::surface_impl>(
                    pNumXPoints, pNumYPoints, pDataType, pMarkerType);
        };
    }

    Surface(const fg_surface pOther)
        : ChartRenderableBase<detail::surface_impl>(
              reinterpret_cast<Surface *>(pOther)->impl()) {}
};

class VectorField : public ChartRenderableBase<detail::vector_field_impl> {
   public:
    VectorField(const unsigned pNumPoints, const theia::dtype pDataType,
                const theia::ChartType pChartType) {
        if (pChartType == FG_CHART_2D) {
            mShrdPtr = std::make_shared<detail::vector_field2d_impl>(pNumPoints,
                                                                     pDataType);
        } else {
            mShrdPtr = std::make_shared<detail::vector_field_impl>(pNumPoints,
                                                                   pDataType);
        }
    }

    VectorField(const fg_vector_field pOther)
        : ChartRenderableBase<detail::vector_field_impl>(
              reinterpret_cast<VectorField *>(pOther)->impl()) {}

    inline unsigned dbo() const { return mShrdPtr->directions(); }

    inline size_t dboSize() const { return mShrdPtr->directionsSize(); }
};

}  // namespace common
}  // namespace theia
