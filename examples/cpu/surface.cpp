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

#include <theia/theia.h>
#define USE_THEIA_CPU_COPY_HELPERS
#include <theia/fg/compute_copy.h>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

using namespace std;

static const float XMIN = -8.0f;
static const float XMAX = 8.0f;
static const float YMIN = -8.0f;
static const float YMAX = 8.0f;

const float DX     = 0.5;
const size_t XSIZE = (size_t)((XMAX - XMIN) / DX);
const size_t YSIZE = (size_t)((YMAX - YMIN) / DX);

void genSurface(float dx, std::vector<float>& vec) {
    vec.clear();
    for (float x = XMIN; x < XMAX; x += dx) {
        for (float y = YMIN; y < YMAX; y += dx) {
            vec.push_back(x);
            vec.push_back(y);
            float z = sqrt(x * x + y * y) + 2.2204e-16f;
            vec.push_back(sin(z) / z);
        }
    }
}

int main(void) {
    /*
     * First theia call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other theia::* object to be created successfully
     */
    theia::Window wnd(1024, 768, "3d Surface Demo");
    wnd.makeCurrent();

    theia::Chart chart(FG_CHART_3D);
    chart.setAxesLimits(XMIN - 2.0f, XMAX + 2.0f, YMIN - 2.0f, YMAX + 2.0f,
                        -0.5f, 1.f);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    theia::Surface surf = chart.surface(XSIZE, YSIZE, theia::f32);
    surf.setColor(FG_YELLOW);

    // generate a surface
    std::vector<float> function;

    genSurface(DX, function);

    GfxHandle* handle;
    createGLBuffer(&handle, surf.vertices(), THEIA_VERTEX_BUFFER);

    /* copy your data into the pixel buffer object exposed by
     * theia::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, theia provides copy headers
     * along with the library to help with this task
     */
    copyToGLBuffer(handle, (ComputeResourceHandle)function.data(),
                   surf.verticesSize());

    do { wnd.draw(chart); } while (!wnd.close());

    releaseGLBuffer(handle);

    return 0;
}
