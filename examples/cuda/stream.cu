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

#include <cuda_runtime.h>
#include <theia/theia.h>
#define USE_THEIA_CUDA_COPY_HELPERS
#include <theia/fg/compute_copy.h>

const unsigned DIMX = 640;
const unsigned DIMY = 480;
const float MINIMUM = 1.0f;
const float MAXIMUM = 20.f;
const float STEP    = 2.0f;
const int NELEMS    = (int)((MAXIMUM - MINIMUM + 1) / STEP);

void generateColors(float* colors);

void generatePoints(float* points, float* dirs);

inline int divup(int a, int b) { return (a + b - 1) / b; }

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

    float* points;
    float* colors;
    float* dirs;

    THEIA_CUDA_CHECK(cudaMalloc((void**)&points, 3 * numElems * sizeof(float)));
    THEIA_CUDA_CHECK(cudaMalloc((void**)&colors, 3 * numElems * sizeof(float)));
    THEIA_CUDA_CHECK(cudaMalloc((void**)&dirs, 3 * numElems * sizeof(float)));

    generatePoints(points, dirs);
    generateColors(colors);

    GfxHandle* handles[3];
    createGLBuffer(&handles[0], field.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[1], field.colors(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[2], field.directions(), THEIA_VERTEX_BUFFER);

    copyToGLBuffer(handles[0], (ComputeResourceHandle)points,
                   field.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)colors,
                   field.colorsSize());
    copyToGLBuffer(handles[2], (ComputeResourceHandle)dirs,
                   field.directionsSize());

    do { wnd.draw(chart); } while (!wnd.close());

    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);

    THEIA_CUDA_CHECK(cudaFree(points));
    THEIA_CUDA_CHECK(cudaFree(colors));
    THEIA_CUDA_CHECK(cudaFree(dirs));

    return 0;
}

__global__ void genColorsKernel(float* colors, int nelems) {
    const float AF_BLUE[4]   = {0.0588f, 0.1137f, 0.2745f, 1.0f};
    const float AF_ORANGE[4] = {0.8588f, 0.6137f, 0.0745f, 1.0f};

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nelems) {
        if (i % 2 == 0) {
            colors[3 * i + 0] = AF_ORANGE[0];
            colors[3 * i + 1] = AF_ORANGE[1];
            colors[3 * i + 2] = AF_ORANGE[2];
        } else {
            colors[3 * i + 0] = AF_BLUE[0];
            colors[3 * i + 1] = AF_BLUE[1];
            colors[3 * i + 2] = AF_BLUE[2];
        }
    }
}

void generateColors(float* colors) {
    const int numElems = NELEMS * NELEMS * NELEMS;
    static const dim3 threads(512);
    dim3 blocks(divup(numElems, threads.x));

    // clang-format off
    genColorsKernel<<<blocks, threads>>>(colors, numElems);
    // clang-format on
}

__global__ void pointGenKernel(float* points, float* dirs, int nBBS0,
                               int nelems, float minimum, float step) {
    int k = blockIdx.x / nBBS0;
    int i = blockDim.x * (blockIdx.x - k * nBBS0) + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < nelems && j < nelems && k < nelems) {
        float x = minimum + i * step;
        float y = minimum + j * step;
        float z = minimum + k * step;

        int id = i + j * nelems + k * nelems * nelems;

        points[3 * id + 0] = x;
        points[3 * id + 1] = y;
        points[3 * id + 2] = z;

        dirs[3 * id + 0] = x - 10.f;
        dirs[3 * id + 1] = y - 10.f;
        dirs[3 * id + 2] = z - 10.f;
    }
}

void generatePoints(float* points, float* dirs) {
    static dim3 threads(8, 8);

    int blk_x = divup(NELEMS, threads.x);
    int blk_y = divup(NELEMS, threads.y);

    dim3 blocks(blk_x * NELEMS, blk_y);

    // clang-format off
    pointGenKernel<<<blocks, threads>>>(points, dirs, blk_x, NELEMS, MINIMUM,
                                        STEP);
    // clang-format on
}
