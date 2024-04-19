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

#include <theia/fg/plot.h>

#include <error.hpp>

#include <utility>

namespace theia {
Plot::Plot(const unsigned pNumPoints, const dtype pDataType,
           const ChartType pChartType, const PlotType pPlotType,
           const MarkerType pMarkerType) {
    fg_plot temp = 0;
    FG_THROW(fg_create_plot(&temp, pNumPoints, (fg_dtype)pDataType, pChartType,
                            pPlotType, pMarkerType));
    std::swap(mValue, temp);
}

Plot::Plot(const Plot& pOther) {
    fg_plot temp = 0;

    FG_THROW(fg_retain_plot(&temp, pOther.get()));

    std::swap(mValue, temp);
}

Plot::Plot(const fg_plot pHandle) : mValue(pHandle) {}

Plot::~Plot() { fg_release_plot(get()); }

void Plot::setColor(const Color pColor) {
    float r = (((int)pColor >> 24) & 0xFF) / 255.f;
    float g = (((int)pColor >> 16) & 0xFF) / 255.f;
    float b = (((int)pColor >> 8) & 0xFF) / 255.f;
    float a = (((int)pColor) & 0xFF) / 255.f;

    FG_THROW(fg_set_plot_color(get(), r, g, b, a));
}

void Plot::setColor(const float pRed, const float pGreen, const float pBlue,
                    const float pAlpha) {
    FG_THROW(fg_set_plot_color(get(), pRed, pGreen, pBlue, pAlpha));
}

void Plot::setLegend(const char* pLegend) {
    FG_THROW(fg_set_plot_legend(get(), pLegend));
}

void Plot::setMarkerSize(const float pMarkerSize) {
    FG_THROW(fg_set_plot_marker_size(get(), pMarkerSize));
}

unsigned Plot::vertices() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_vertex_buffer(&temp, get()));
    return temp;
}

unsigned Plot::colors() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_color_buffer(&temp, get()));
    return temp;
}

unsigned Plot::alphas() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_alpha_buffer(&temp, get()));
    return temp;
}

unsigned Plot::radii() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_radii_buffer(&temp, get()));
    return temp;
}

unsigned Plot::verticesSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_vertex_buffer_size(&temp, get()));
    return temp;
}

unsigned Plot::colorsSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_color_buffer_size(&temp, get()));
    return temp;
}

unsigned Plot::alphasSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_alpha_buffer_size(&temp, get()));
    return temp;
}

unsigned Plot::radiiSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_plot_radii_buffer_size(&temp, get()));
    return temp;
}

fg_plot Plot::get() const { return mValue; }
}  // namespace theia
