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

#include <theia/fg/vector_field.h>

#include <error.hpp>

#include <utility>

namespace theia {
VectorField::VectorField(const unsigned pNumPoints, const dtype pDataType,
                         const ChartType pChartType) {
    fg_vector_field temp = 0;
    FG_THROW(fg_create_vector_field(&temp, pNumPoints, (fg_dtype)pDataType,
                                    pChartType));
    std::swap(mValue, temp);
}

VectorField::VectorField(const VectorField& pOther) {
    fg_vector_field temp = 0;

    FG_THROW(fg_retain_vector_field(&temp, pOther.get()));

    std::swap(mValue, temp);
}

VectorField::VectorField(const fg_vector_field pHandle) : mValue(pHandle) {}

VectorField::~VectorField() { fg_release_vector_field(get()); }

void VectorField::setColor(const Color pColor) {
    float r = (((int)pColor >> 24) & 0xFF) / 255.f;
    float g = (((int)pColor >> 16) & 0xFF) / 255.f;
    float b = (((int)pColor >> 8) & 0xFF) / 255.f;
    float a = (((int)pColor) & 0xFF) / 255.f;

    FG_THROW(fg_set_vector_field_color(get(), r, g, b, a));
}

void VectorField::setColor(const float pRed, const float pGreen,
                           const float pBlue, const float pAlpha) {
    FG_THROW(fg_set_vector_field_color(get(), pRed, pGreen, pBlue, pAlpha));
}

void VectorField::setLegend(const char* pLegend) {
    FG_THROW(fg_set_vector_field_legend(get(), pLegend));
}

unsigned VectorField::vertices() const {
    unsigned temp = 0;
    FG_THROW(fg_get_vector_field_vertex_buffer(&temp, get()));
    return temp;
}

unsigned VectorField::colors() const {
    unsigned temp = 0;
    FG_THROW(fg_get_vector_field_color_buffer(&temp, get()));
    return temp;
}

unsigned VectorField::alphas() const {
    unsigned temp = 0;
    FG_THROW(fg_get_vector_field_alpha_buffer(&temp, get()));
    return temp;
}

unsigned VectorField::directions() const {
    unsigned temp = 0;
    FG_THROW(fg_get_vector_field_direction_buffer(&temp, get()));
    return temp;
}

unsigned VectorField::verticesSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_vector_field_vertex_buffer_size(&temp, get()));
    return temp;
}

unsigned VectorField::colorsSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_vector_field_color_buffer_size(&temp, get()));
    return temp;
}

unsigned VectorField::alphasSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_vector_field_alpha_buffer_size(&temp, get()));
    return temp;
}

unsigned VectorField::directionsSize() const {
    unsigned temp = 0;
    FG_THROW(fg_get_vector_field_direction_buffer_size(&temp, get()));
    return temp;
}

fg_vector_field VectorField::get() const { return mValue; }
}  // namespace theia
