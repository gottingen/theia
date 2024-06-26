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

const unsigned DIMX      = 640;
const unsigned DIMY      = 480;
const float PI           = 3.14159265359f;
const float MINIMUM      = 1.0f;
const float MAXIMUM      = 20.f;
const float STEP         = 2.0f;
const float NELEMS       = (MAXIMUM - MINIMUM + 1) / STEP;
const unsigned DPOINTS[] = {5, 5, 5, 15, 15, 5, 15, 15};

using namespace std;

void generatePoints(std::vector<float> &points, std::vector<float> &dirs) {
    points.clear();

    for (int j = 0; j < NELEMS; ++j) {
        float y = MINIMUM + j * STEP;
        for (int i = 0; i < NELEMS; ++i) {
            float x = MINIMUM + i * STEP;
            points.push_back(x);
            points.push_back(y);
            dirs.push_back(sin(2 * PI * x / 10.f));
            dirs.push_back(sin(2 * PI * y / 10.f));
        }
    }
}

int main(void) {
    /*
     * First theia call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other theia::* object to be created successfully
     */
    theia::Window wnd(DIMX, DIMY, "Vector Field Demo");
    wnd.makeCurrent();

    theia::Chart chart(FG_CHART_2D);
    chart.setAxesLimits(MINIMUM - 1.0f, MAXIMUM, MINIMUM - 1.0f, MAXIMUM);
    chart.setAxesTitles("x-axis", "y-axis");

    theia::Plot divPoints =
        chart.plot(4, theia::u32, FG_PLOT_SCATTER, FG_MARKER_CIRCLE);
    divPoints.setColor(0.9f, 0.9f, 0.0f, 1.f);
    divPoints.setLegend("Convergence Points");
    divPoints.setMarkerSize(24);

    theia::VectorField field =
        chart.vectorField((unsigned)(NELEMS * NELEMS), theia::f32);
    field.setColor(0.f, 0.6f, 0.3f, 1.f);

    std::vector<float> points;
    std::vector<float> dirs;
    generatePoints(points, dirs);

    GfxHandle *handles[3];

    createGLBuffer(&handles[0], divPoints.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[1], field.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[2], field.directions(), THEIA_VERTEX_BUFFER);

    copyToGLBuffer(handles[0], (ComputeResourceHandle)DPOINTS,
                   divPoints.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)points.data(),
                   field.verticesSize());
    copyToGLBuffer(handles[2], (ComputeResourceHandle)dirs.data(),
                   field.directionsSize());

    do { wnd.draw(chart); } while (!wnd.close());

    // destroy GL-cpu interop buffers
    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);

    return 0;
}
