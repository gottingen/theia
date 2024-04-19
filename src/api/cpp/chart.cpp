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

#include <error.hpp>
#include <theia/fg/chart.h>
#include <theia/fg/font.h>
#include <theia/fg/histogram.h>
#include <theia/fg/image.h>
#include <theia/fg/plot.h>
#include <theia/fg/surface.h>
#include <theia/fg/window.h>

#include <utility>

namespace theia {

Chart::Chart(const ChartType cType) {
    fg_chart temp = 0;
    FG_THROW(fg_create_chart(&temp, (fg_chart_type)cType));
    std::swap(mValue, temp);
}

Chart::Chart(const Chart& pOther) {
    fg_chart temp = 0;
    FG_THROW(fg_retain_chart(&temp, pOther.get()));
    std::swap(mValue, temp);
}

Chart::~Chart() { fg_release_chart(get()); }

void Chart::setAxesTitles(const char* pX, const char* pY, const char* pZ) {
    FG_THROW(fg_set_chart_axes_titles(get(), pX, pY, pZ));
}

void Chart::setAxesLimits(const float pXmin, const float pXmax,
                          const float pYmin, const float pYmax,
                          const float pZmin, const float pZmax) {
    FG_THROW(fg_set_chart_axes_limits(get(), pXmin, pXmax, pYmin, pYmax, pZmin,
                                      pZmax));
}

void Chart::setAxesLabelFormat(const char* pXFormat, const char* pYFormat,
                               const char* pZFormat) {
    FG_THROW(fg_set_chart_label_format(get(), pXFormat, pYFormat, pZFormat));
}

void Chart::getAxesLimits(float* pXmin, float* pXmax, float* pYmin,
                          float* pYmax, float* pZmin, float* pZmax) {
    FG_THROW(fg_get_chart_axes_limits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax,
                                      get()));
}

void Chart::setLegendPosition(const float pX, const float pY) {
    FG_THROW(fg_set_chart_legend_position(get(), pX, pY));
}

void Chart::add(const Image& pImage) {
    FG_THROW(fg_append_image_to_chart(get(), pImage.get()));
}

void Chart::add(const Histogram& pHistogram) {
    FG_THROW(fg_append_histogram_to_chart(get(), pHistogram.get()));
}

void Chart::add(const Plot& pPlot) {
    FG_THROW(fg_append_plot_to_chart(get(), pPlot.get()));
}

void Chart::add(const Surface& pSurface) {
    FG_THROW(fg_append_surface_to_chart(get(), pSurface.get()));
}

void Chart::add(const VectorField& pVectorField) {
    FG_THROW(fg_append_vector_field_to_chart(get(), pVectorField.get()));
}

void Chart::remove(const Image& pImage) {
    FG_THROW(fg_remove_image_from_chart(get(), pImage.get()));
}

void Chart::remove(const Histogram& pHistogram) {
    FG_THROW(fg_remove_histogram_from_chart(get(), pHistogram.get()));
}

void Chart::remove(const Plot& pPlot) {
    FG_THROW(fg_remove_plot_from_chart(get(), pPlot.get()));
}

void Chart::remove(const Surface& pSurface) {
    FG_THROW(fg_remove_surface_from_chart(get(), pSurface.get()));
}

void Chart::remove(const VectorField& pVectorField) {
    FG_THROW(fg_remove_vector_field_from_chart(get(), pVectorField.get()));
}

Image Chart::image(const unsigned pWidth, const unsigned pHeight,
                   const ChannelFormat pFormat, const dtype pDataType) {
    fg_image temp = 0;
    FG_THROW(fg_add_image_to_chart(&temp, get(), pWidth, pHeight,
                                   (fg_channel_format)pFormat,
                                   (fg_dtype)pDataType));
    return Image(temp);
}

Histogram Chart::histogram(const unsigned pNBins, const dtype pDataType) {
    fg_histogram temp = 0;
    FG_THROW(
        fg_add_histogram_to_chart(&temp, get(), pNBins, (fg_dtype)pDataType));
    return Histogram(temp);
}

Plot Chart::plot(const unsigned pNumPoints, const dtype pDataType,
                 const PlotType pPlotType, const MarkerType pMarkerType) {
    fg_plot temp = 0;
    FG_THROW(fg_add_plot_to_chart(&temp, get(), pNumPoints, (fg_dtype)pDataType,
                                  pPlotType, pMarkerType));
    return Plot(temp);
}

Surface Chart::surface(const unsigned pNumXPoints, const unsigned pNumYPoints,
                       const dtype pDataType, const PlotType pPlotType,
                       const MarkerType pMarkerType) {
    fg_surface temp = 0;
    FG_THROW(fg_add_surface_to_chart(&temp, get(), pNumXPoints, pNumYPoints,
                                     (fg_dtype)pDataType, pPlotType,
                                     pMarkerType));
    return Surface(temp);
}

VectorField Chart::vectorField(const unsigned pNumPoints,
                               const dtype pDataType) {
    fg_vector_field temp = 0;
    FG_THROW(fg_add_vector_field_to_chart(&temp, get(), pNumPoints,
                                          (fg_dtype)pDataType));
    return VectorField(temp);
}

void Chart::render(const Window& pWindow, const int pX, const int pY,
                   const int pVPW, const int pVPH) const {
    FG_THROW(fg_render_chart(pWindow.get(), get(), pX, pY, pVPW, pVPH));
}

fg_chart Chart::get() const { return mValue; }

ChartType Chart::getChartType() const {
    fg_chart_type retVal = (fg_chart_type)0;
    FG_THROW(fg_get_chart_type(&retVal, get()));
    return retVal;
}

}  // namespace theia
