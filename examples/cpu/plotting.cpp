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
#include <cstdio>
#include <iostream>
#include <vector>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

const float FRANGE_START = 0.f;
const float FRANGE_END   = 2.f * 3.1415926f;

using namespace std;
void map_range_to_vec_vbo(float range_start, float range_end, float dx,
                          std::vector<float>& vec, float (*map)(float)) {
    if (range_start > range_end && dx > 0) return;
    for (float i = range_start; i < range_end; i += dx) {
        vec.push_back(i);
        vec.push_back((*map)(i));
    }
}

int main(void) {
    std::vector<float> sinData;
    std::vector<float> cosData;
    std::vector<float> tanData;
    std::vector<float> logData;
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, sinData, &sinf);
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, cosData, &cosf);
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, tanData, &tanf);
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, logData, &log10f);

    /*
     * First theia call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other theia::* object to be created successfully
     */
    theia::Window wnd(DIMX, DIMY, "Chart with Sin, Cos, Tan and Log10 Plots");
    wnd.makeCurrent();

    theia::Chart chart(FG_CHART_2D);
    chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.0f, 1.0f);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    theia::Plot plt0 = chart.plot((unsigned)(sinData.size() / 2),
                                  theia::f32);  // create a default plot
    theia::Plot plt1 =
        chart.plot((unsigned)(cosData.size() / 2), theia::f32, FG_PLOT_LINE,
                   FG_MARKER_NONE);  // or specify a specific plot type
    theia::Plot plt2 = chart.plot(
        (unsigned)(tanData.size() / 2), theia::f32, FG_PLOT_LINE,
        FG_MARKER_TRIANGLE);  // last parameter specifies marker shape
    theia::Plot plt3 = chart.plot((unsigned)(logData.size() / 2), theia::f32,
                                  FG_PLOT_SCATTER, FG_MARKER_CROSS);

    /*
     * Set plot colors
     */
    plt0.setColor(FG_RED);
    plt1.setColor(FG_BLUE);
    plt2.setColor(FG_YELLOW);                 // use a theia predefined color
    plt3.setColor((theia::Color)0x257973FF);  // or any hex-valued color
    /*
     * Set plot legends
     */
    plt0.setLegend("Sine");
    plt1.setLegend("Cosine");
    plt2.setLegend("Tangent");
    plt3.setLegend("Log base 10");

    GfxHandle* handles[4];
    createGLBuffer(&handles[0], plt0.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[1], plt1.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[2], plt2.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[3], plt3.vertices(), THEIA_VERTEX_BUFFER);

    /* copy your data into the pixel buffer object exposed by
     * theia::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, theia provides copy headers
     * along with the library to help with this task
     */
    copyToGLBuffer(handles[0], (ComputeResourceHandle)sinData.data(),
                   plt0.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)cosData.data(),
                   plt1.verticesSize());
    copyToGLBuffer(handles[2], (ComputeResourceHandle)tanData.data(),
                   plt2.verticesSize());
    copyToGLBuffer(handles[3], (ComputeResourceHandle)logData.data(),
                   plt3.verticesSize());

    do { wnd.draw(chart); } while (!wnd.close());

    printf("Removed Log base 10 plot from chart\n");
    chart.remove(plt3);
    releaseGLBuffer(handles[3]);

    printf("Rendering Sine, Cosine and Tangent Again ...\n");
    wnd.setTitle("Chart with Sin, Cos and Tan Plots");
    do { wnd.draw(chart); } while (!wnd.close());

    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);

    return 0;
}
