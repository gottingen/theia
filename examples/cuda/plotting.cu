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

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <theia/theia.h>
#define USE_THEIA_CUDA_COPY_HELPERS
#include <theia/fg/compute_copy.h>
#include <cstdio>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float dx           = 0.1f;
static const float FRANGE_START = 0.f;
static const float FRANGE_END   = 2 * 3.141592f;
static const size_t DATA_SIZE   = (size_t)((FRANGE_END - FRANGE_START) / dx);

void kernel(float* dev_out, int functionCode);

int main(void) {
    float* sin_out;
    float* cos_out;
    float* tan_out;
    float* log_out;

    /*
     * First theia call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other theia::* object to be created successfully
     */
    theia::Window wnd(DIMX, DIMY, "Plotting Demo");
    wnd.makeCurrent();

    theia::Chart chart(FG_CHART_2D);
    chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.0f, 1.0f);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    theia::Plot plt0 =
        chart.plot(DATA_SIZE, theia::f32);  // create a default plot
    theia::Plot plt1 =
        chart.plot(DATA_SIZE, theia::f32, FG_PLOT_LINE,
                   FG_MARKER_NONE);  // or specify a specific plot type
    theia::Plot plt2 = chart.plot(
        DATA_SIZE, theia::f32, FG_PLOT_LINE,
        FG_MARKER_TRIANGLE);  // last parameter specifies marker shape
    theia::Plot plt3 =
        chart.plot(DATA_SIZE, theia::f32, FG_PLOT_SCATTER, FG_MARKER_CROSS);

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

    THEIA_CUDA_CHECK(
        cudaMalloc((void**)&sin_out, sizeof(float) * DATA_SIZE * 2));
    THEIA_CUDA_CHECK(
        cudaMalloc((void**)&cos_out, sizeof(float) * DATA_SIZE * 2));
    THEIA_CUDA_CHECK(
        cudaMalloc((void**)&tan_out, sizeof(float) * DATA_SIZE * 2));
    THEIA_CUDA_CHECK(
        cudaMalloc((void**)&log_out, sizeof(float) * DATA_SIZE * 2));

    kernel(sin_out, 0);
    kernel(cos_out, 1);
    kernel(tan_out, 2);
    kernel(log_out, 3);

    GfxHandle* handles[4];
    createGLBuffer(&handles[0], plt0.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[1], plt1.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[2], plt2.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[3], plt3.vertices(), THEIA_VERTEX_BUFFER);

    /* copy your data into the vertex buffer object exposed by
     * theia::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, theia provides copy headers
     * along with the library to help with this task
     */
    copyToGLBuffer(handles[0], (ComputeResourceHandle)sin_out,
                   plt0.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)cos_out,
                   plt1.verticesSize());
    copyToGLBuffer(handles[2], (ComputeResourceHandle)tan_out,
                   plt2.verticesSize());
    copyToGLBuffer(handles[3], (ComputeResourceHandle)log_out,
                   plt3.verticesSize());

    do { wnd.draw(chart); } while (!wnd.close());

    THEIA_CUDA_CHECK(cudaFree(sin_out));
    THEIA_CUDA_CHECK(cudaFree(cos_out));
    THEIA_CUDA_CHECK(cudaFree(tan_out));
    THEIA_CUDA_CHECK(cudaFree(log_out));
    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);
    releaseGLBuffer(handles[3]);

    return 0;
}

__global__ void simple_sinf(float* out, const size_t _data_size, int fnCode,
                            const float _dx, const float _frange_start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _data_size) {
        float x  = _frange_start + i * _dx;
        int idx  = 2 * i;
        out[idx] = x;

        switch (fnCode) {
            case 0: out[idx + 1] = sinf(x); break;
            case 1: out[idx + 1] = cosf(x); break;
            case 2: out[idx + 1] = tanf(x); break;
            case 3: out[idx + 1] = log10f(x); break;
        }
    }
}

inline int divup(int a, int b) { return (a + b - 1) / b; }

void kernel(float* dev_out, int functionCode) {
    static const dim3 threads(1024);
    dim3 blocks(divup(DATA_SIZE, 1024));

    // clang-format off
    simple_sinf<<<blocks, threads>>>(dev_out, DATA_SIZE, functionCode, dx,
                                     FRANGE_START);
    // clang-format on
}
