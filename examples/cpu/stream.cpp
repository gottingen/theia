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

const unsigned DIMX = 640;
const unsigned DIMY = 480;
const float PI      = 3.14159265359f;
const float MINIMUM = 1.0f;
const float MAXIMUM = 20.f;
const float STEP    = 2.0f;
const int NELEMS    = (int)((MAXIMUM - MINIMUM + 1) / STEP);

using namespace std;

void generateColors(std::vector<float>& colors) {
    static const float AF_BLUE[]   = {0.0588f, 0.1137f, 0.2745f, 1.0f};
    static const float AF_ORANGE[] = {0.8588f, 0.6137f, 0.0745f, 1.0f};

    int numElems = NELEMS * NELEMS * NELEMS;
    colors.clear();
    for (int i = 0; i < numElems; ++i) {
        if ((i % 2) == 0) {
            colors.push_back(AF_ORANGE[0]);
            colors.push_back(AF_ORANGE[1]);
            colors.push_back(AF_ORANGE[2]);
        } else {
            colors.push_back(AF_BLUE[0]);
            colors.push_back(AF_BLUE[1]);
            colors.push_back(AF_BLUE[2]);
        }
    }
}

void generatePoints(std::vector<float>& points, std::vector<float>& dirs) {
    points.clear();

    for (int k = 0; k < NELEMS; ++k) {
        float z = MINIMUM + k * STEP;
        for (int j = 0; j < NELEMS; ++j) {
            float y = MINIMUM + j * STEP;
            for (int i = 0; i < NELEMS; ++i) {
                float x = MINIMUM + i * STEP;
                points.push_back(x);
                points.push_back(y);
                points.push_back(z);
                dirs.push_back(x - 10.0f);
                dirs.push_back(y - 10.0f);
                dirs.push_back(z - 10.0f);
            }
        }
    }
}

int main(void) {
    /*
     * First theia call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other theia::* object to be created successfully
     */
    theia::Window wnd(DIMX, DIMY, "3D Vector Field Demo");
    wnd.makeCurrent();

    theia::Chart chart(FG_CHART_3D);
    chart.setAxesLimits(MINIMUM - 1.0f, MAXIMUM, MINIMUM - 1.0f, MAXIMUM,
                        MINIMUM - 1.0f, MAXIMUM);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    int numElems             = NELEMS * NELEMS * NELEMS;
    theia::VectorField field = chart.vectorField(numElems, theia::f32);
    field.setColor(0.f, 1.f, 0.f, 1.f);

    std::vector<float> points;
    std::vector<float> colors;
    std::vector<float> dirs;
    generatePoints(points, dirs);
    generateColors(colors);

    GfxHandle* handles[3];
    createGLBuffer(&handles[0], field.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[1], field.colors(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[2], field.directions(), THEIA_VERTEX_BUFFER);

    copyToGLBuffer(handles[0], (ComputeResourceHandle)points.data(),
                   field.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)colors.data(),
                   field.colorsSize());
    copyToGLBuffer(handles[2], (ComputeResourceHandle)dirs.data(),
                   field.directionsSize());

    do { wnd.draw(chart); } while (!wnd.close());

    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);

    return 0;
}
