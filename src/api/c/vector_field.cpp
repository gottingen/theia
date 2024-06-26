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

#include <common/chart_renderables.hpp>
#include <common/handle.hpp>
#include <theia/fg/vector_field.h>

using namespace theia;

using theia::common::getVectorField;

fg_err fg_create_vector_field(fg_vector_field* pField, const unsigned pNPoints,
                              const fg_dtype pType,
                              const fg_chart_type pChartType) {
    try {
        ARG_ASSERT(1, (pNPoints > 0));

        *pField = getHandle(
            new common::VectorField(pNPoints, (theia::dtype)pType, pChartType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_vector_field(fg_vector_field* pOut, fg_vector_field pIn) {
    try {
        ARG_ASSERT(1, (pIn != 0));

        common::VectorField* temp =
            new common::VectorField(getVectorField(pIn));
        *pOut = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_vector_field(fg_vector_field pField) {
    try {
        ARG_ASSERT(0, (pField != 0));

        delete getVectorField(pField);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_vector_field_color(fg_vector_field pField, const float pRed,
                                 const float pGreen, const float pBlue,
                                 const float pAlpha) {
    try {
        ARG_ASSERT(0, (pField != 0));

        getVectorField(pField)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_vector_field_legend(fg_vector_field pField, const char* pLegend) {
    try {
        ARG_ASSERT(0, (pField != 0));
        ARG_ASSERT(1, (pLegend != 0));

        getVectorField(pField)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_vertex_buffer(unsigned* pOut,
                                         const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = getVectorField(pField)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_color_buffer(unsigned* pOut,
                                        const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = getVectorField(pField)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_alpha_buffer(unsigned* pOut,
                                        const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = getVectorField(pField)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_direction_buffer(unsigned* pOut,
                                            const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = getVectorField(pField)->dbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_vertex_buffer_size(unsigned* pOut,
                                              const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = (unsigned)getVectorField(pField)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_color_buffer_size(unsigned* pOut,
                                             const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = (unsigned)getVectorField(pField)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_alpha_buffer_size(unsigned* pOut,
                                             const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = (unsigned)getVectorField(pField)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_direction_buffer_size(unsigned* pOut,
                                                 const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = (unsigned)getVectorField(pField)->dboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}
