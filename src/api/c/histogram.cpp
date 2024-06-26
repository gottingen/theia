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
#include <theia/fg/histogram.h>

using namespace theia;
using theia::common::getHistogram;

fg_err fg_create_histogram(fg_histogram* pHistogram, const unsigned pNBins,
                           const fg_dtype pType) {
    try {
        ARG_ASSERT(1, (pNBins > 0));

        *pHistogram =
            getHandle(new common::Histogram(pNBins, (theia::dtype)pType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_histogram(fg_histogram* pOut, fg_histogram pIn) {
    try {
        ARG_ASSERT(1, (pIn != 0));

        common::Histogram* temp = new common::Histogram(getHistogram(pIn));
        *pOut                   = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_histogram(fg_histogram pHistogram) {
    try {
        ARG_ASSERT(0, (pHistogram != 0));

        delete getHistogram(pHistogram);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_histogram_color(fg_histogram pHistogram, const float pRed,
                              const float pGreen, const float pBlue,
                              const float pAlpha) {
    try {
        ARG_ASSERT(0, (pHistogram != 0));

        getHistogram(pHistogram)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_histogram_legend(fg_histogram pHistogram, const char* pLegend) {
    try {
        ARG_ASSERT(0, (pHistogram != 0));
        ARG_ASSERT(1, (pLegend != 0));

        getHistogram(pHistogram)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_vertex_buffer(unsigned* pOut,
                                      const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = getHistogram(pHistogram)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_color_buffer(unsigned* pOut,
                                     const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = getHistogram(pHistogram)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_alpha_buffer(unsigned* pOut,
                                     const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = getHistogram(pHistogram)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_vertex_buffer_size(unsigned* pOut,
                                           const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = (unsigned)getHistogram(pHistogram)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_color_buffer_size(unsigned* pOut,
                                          const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = (unsigned)getHistogram(pHistogram)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_alpha_buffer_size(unsigned* pOut,
                                          const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = (unsigned)getHistogram(pHistogram)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}
