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

#include <common/chart.hpp>
#include <common/chart_renderables.hpp>
#include <common/font.hpp>
#include <common/image.hpp>
#include <common/window.hpp>
#include <theia/fg/exception.h>

namespace theia {
namespace common {

fg_window getHandle(Window* pValue);

fg_font getHandle(Font* pValue);

fg_image getHandle(Image* pValue);

fg_chart getHandle(Chart* pValue);

fg_histogram getHandle(Histogram* pValue);

fg_plot getHandle(Plot* pValue);

fg_surface getHandle(Surface* pValue);

fg_vector_field getHandle(VectorField* pValue);

Window* getWindow(const fg_window& pValue);

Font* getFont(const fg_font& pValue);

Image* getImage(const fg_image& pValue);

Chart* getChart(const fg_chart& pValue);

Histogram* getHistogram(const fg_histogram& pValue);

Plot* getPlot(const fg_plot& pValue);

Surface* getSurface(const fg_surface& pValue);

VectorField* getVectorField(const fg_vector_field& pValue);

}  // namespace common
}  // namespace theia
