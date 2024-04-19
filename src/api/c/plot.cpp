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
#include <theia/fg/plot.h>

using namespace theia;

using theia::common::getPlot;

fg_err fg_create_plot(fg_plot* pPlot, const unsigned pNPoints,
                      const fg_dtype pType, const fg_chart_type pChartType,
                      const fg_plot_type pPlotType,
                      const fg_marker_type pMarkerType) {
    try {
        ARG_ASSERT(1, (pNPoints > 0));

        *pPlot = getHandle(new common::Plot(
            pNPoints, (theia::dtype)pType, pPlotType, pMarkerType, pChartType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_plot(fg_plot* pOut, fg_plot pIn) {
    try {
        ARG_ASSERT(1, (pIn != 0));

        common::Plot* temp = new common::Plot(getPlot(pIn));
        *pOut              = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_plot(fg_plot pPlot) {
    try {
        ARG_ASSERT(0, (pPlot != 0));

        delete getPlot(pPlot);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_plot_color(fg_plot pPlot, const float pRed, const float pGreen,
                         const float pBlue, const float pAlpha) {
    try {
        ARG_ASSERT(0, (pPlot != 0));

        getPlot(pPlot)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_plot_legend(fg_plot pPlot, const char* pLegend) {
    try {
        ARG_ASSERT(0, (pPlot != 0));
        ARG_ASSERT(1, (pLegend != 0));

        getPlot(pPlot)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_plot_marker_size(fg_plot pPlot, const float pMarkerSize) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        getPlot(pPlot)->setMarkerSize(pMarkerSize);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_vertex_buffer(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = getPlot(pPlot)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_color_buffer(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = getPlot(pPlot)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_alpha_buffer(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = getPlot(pPlot)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_radii_buffer(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = getPlot(pPlot)->mbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_vertex_buffer_size(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = (unsigned)getPlot(pPlot)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_color_buffer_size(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = (unsigned)getPlot(pPlot)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_alpha_buffer_size(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = (unsigned)getPlot(pPlot)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_radii_buffer_size(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = (unsigned)getPlot(pPlot)->mboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}
