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

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float ZMIN = 0.1f;
static const float ZMAX = 10.f;

const float DX     = 0.005f;
const size_t ZSIZE = (size_t)((ZMAX - ZMIN) / DX + 1);

using namespace std;

void generateCurve(float t, float dx, std::vector<float>& vec) {
    vec.clear();
    for (int i = 0; i < (int)ZSIZE; ++i) {
        float z = ZMIN + i * dx;
        vec.push_back((float)(cos(z * t + t) / z));
        vec.push_back((float)(sin(z * t + t) / z));
        vec.push_back((float)(z + 0.1 * sin(t)));
    }
}

int main(void) {
    /*
     * First theia call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other theia::* object to be created successfully
     */
    theia::Window wnd(DIMX, DIMY, "Three dimensional line plot demo");
    wnd.makeCurrent();

    theia::Chart chart(FG_CHART_3D);

    chart.setAxesLabelFormat("%3.1f", "%3.1f", "%.2e");

    chart.setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, 0.f, 10.f);

    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    theia::Plot plot3 = chart.plot(ZSIZE, theia::f32);

    // generate a surface
    std::vector<float> function;
    static float t = 0;
    generateCurve(t, DX, function);

    GfxHandle* handle;
    createGLBuffer(&handle, plot3.vertices(), THEIA_VERTEX_BUFFER);

    /* copy your data into the pixel buffer object exposed by
     * theia::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, theia provides copy headers
     * along with the library to help with this task
     */
    copyToGLBuffer(handle, (ComputeResourceHandle)function.data(),
                   plot3.verticesSize());

    do {
        t += 0.01f;
        generateCurve(t, DX, function);
        copyToGLBuffer(handle, (ComputeResourceHandle)function.data(),
                       plot3.verticesSize());
        wnd.draw(chart);
    } while (!wnd.close());

    releaseGLBuffer(handle);

    return 0;
}
