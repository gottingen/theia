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

static const float ZMIN = 0.1f;
static const float ZMAX = 10.f;

const float DX     = 0.005f;
const size_t ZSIZE = (size_t)((ZMAX - ZMIN) / DX + 1);

void kernel(float t, float dx, float* dev_out);

int main(void) {
    float* dev_out;

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

    static float t = 0;
    THEIA_CUDA_CHECK(cudaMalloc((void**)&dev_out, ZSIZE * 3 * sizeof(float)));
    kernel(t, DX, dev_out);

    GfxHandle* handle;
    createGLBuffer(&handle, plot3.vertices(), THEIA_VERTEX_BUFFER);

    /* copy your data into the vertex buffer object exposed by
     * theia::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, theia provides copy headers
     * along with the library to help with this task
     */
    copyToGLBuffer(handle, (ComputeResourceHandle)dev_out,
                   plot3.verticesSize());

    do {
        t += 0.01f;
        kernel(t, DX, dev_out);
        copyToGLBuffer(handle, (ComputeResourceHandle)dev_out,
                       plot3.verticesSize());
        wnd.draw(chart);
    } while (!wnd.close());

    THEIA_CUDA_CHECK(cudaFree(dev_out));
    releaseGLBuffer(handle);
    return 0;
}

__global__ void generateCurve(float t, float dx, float* out, const float ZMIN,
                              const size_t ZSIZE) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    float z = ZMIN + offset * dx;
    if (offset < ZSIZE) {
        out[3 * offset]     = cos(z * t + t) / z;
        out[3 * offset + 1] = sin(z * t + t) / z;
        out[3 * offset + 2] = z + 0.1 * sin(t);
    }
}

inline int divup(int a, int b) { return (a + b - 1) / b; }

void kernel(float t, float dx, float* dev_out) {
    static const dim3 threads(1024);
    dim3 blocks(divup(ZSIZE, 1024));

    // clang-format off
    generateCurve<<<blocks, threads>>>(t, dx, dev_out, ZMIN, ZSIZE);
    // clang-format on
}
