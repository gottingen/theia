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

/**

\defgroup c_api_functions C API functions
Categorically divided into groups based on the renderable
they are related to.

\addtogroup c_api_functions
@{
    \defgroup chart_functions Chart
    \defgroup font_functions Font
    \defgroup hist_functions Histogram
    \defgroup image_functions Image
    \defgroup plot_functions Plot
    \defgroup surf_functions Surface
    \defgroup util_functions Utility & Helper Functions
    \defgroup vfield_functions Vector Field
    \defgroup win_functions Window

@}

*/

#include <theia/fg/defines.h>
#include <theia/fg/update_buffer.h>
#include <theia/fg/exception.h>
#include <theia/fg/window.h>
#include <theia/fg/font.h>
#include <theia/fg/image.h>
#include <theia/fg/version.h>
#include <theia/fg/plot.h>
#include <theia/fg/surface.h>
#include <theia/fg/histogram.h>
