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

#include <common/err_handling.hpp>
#include <common/handle.hpp>

namespace theia {
namespace common {

fg_window getHandle(Window* pValue) {
    return reinterpret_cast<fg_window>(pValue);
}

fg_font getHandle(Font* pValue) { return reinterpret_cast<fg_font>(pValue); }

fg_image getHandle(Image* pValue) { return reinterpret_cast<fg_image>(pValue); }

fg_chart getHandle(Chart* pValue) { return reinterpret_cast<fg_chart>(pValue); }

fg_histogram getHandle(Histogram* pValue) {
    return reinterpret_cast<fg_histogram>(pValue);
}

fg_plot getHandle(Plot* pValue) { return reinterpret_cast<fg_plot>(pValue); }

fg_surface getHandle(Surface* pValue) {
    return reinterpret_cast<fg_surface>(pValue);
}

fg_vector_field getHandle(VectorField* pValue) {
    return reinterpret_cast<fg_vector_field>(pValue);
}

Window* getWindow(const fg_window& pValue) {
    return reinterpret_cast<common::Window*>(pValue);
}

Font* getFont(const fg_font& pValue) {
    return reinterpret_cast<common::Font*>(pValue);
}

Image* getImage(const fg_image& pValue) {
    return reinterpret_cast<common::Image*>(pValue);
}

Chart* getChart(const fg_chart& pValue) {
    return reinterpret_cast<common::Chart*>(pValue);
}

Histogram* getHistogram(const fg_histogram& pValue) {
    return reinterpret_cast<common::Histogram*>(pValue);
}

Plot* getPlot(const fg_plot& pValue) {
    return reinterpret_cast<common::Plot*>(pValue);
}

Surface* getSurface(const fg_surface& pValue) {
    return reinterpret_cast<common::Surface*>(pValue);
}

VectorField* getVectorField(const fg_vector_field& pValue) {
    return reinterpret_cast<common::VectorField*>(pValue);
}

}  // namespace common
}  // namespace theia
