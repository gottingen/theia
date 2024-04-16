/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cuda_runtime.h>
#include <theia/theia.h>
#define USE_THEIA_CUDA_COPY_HELPERS
#include <theia/fg/compute_copy.h>

#define PI 3.14159265359

const unsigned DIMX      = 640;
const unsigned DIMY      = 480;
const float MINIMUM      = 1.0f;
const float MAXIMUM      = 20.f;
const float STEP         = 2.0f;
const float NELEMS       = (MAXIMUM - MINIMUM + 1) / STEP;
const unsigned DPOINTS[] = {5, 5, 5, 15, 15, 5, 15, 15};

void generatePoints(float* points, float* dirs);

inline int divup(int a, int b) { return (a + b - 1) / b; }

int main(void) {
    unsigned* dpoints;
    float* points;
    float* dirs;
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

    size_t npoints = (size_t)(NELEMS * NELEMS);

    theia::VectorField field =
        chart.vectorField((unsigned)(npoints), theia::f32);
    field.setColor(0.f, 0.6f, 0.3f, 1.f);

    THEIA_CUDA_CHECK(cudaMalloc((void**)&dpoints, 8 * sizeof(unsigned)));
    THEIA_CUDA_CHECK(cudaMalloc((void**)&points, 2 * npoints * sizeof(float)));
    THEIA_CUDA_CHECK(cudaMalloc((void**)&dirs, 2 * npoints * sizeof(float)));

    GfxHandle* handles[3];

    createGLBuffer(&handles[0], divPoints.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[1], field.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[2], field.directions(), THEIA_VERTEX_BUFFER);

    THEIA_CUDA_CHECK(cudaMemcpy(dpoints, DPOINTS, 8 * sizeof(unsigned),
                                cudaMemcpyHostToDevice));
    generatePoints(points, dirs);

    copyToGLBuffer(handles[0], (ComputeResourceHandle)dpoints,
                   divPoints.verticesSize());

    copyToGLBuffer(handles[1], (ComputeResourceHandle)points,
                   field.verticesSize());
    copyToGLBuffer(handles[2], (ComputeResourceHandle)dirs,
                   field.directionsSize());

    do { wnd.draw(chart); } while (!wnd.close());

    // destroy GL-CUDA interop buffers
    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);
    // destroy CUDA handles
    THEIA_CUDA_CHECK(cudaFree(dpoints));
    THEIA_CUDA_CHECK(cudaFree(points));
    THEIA_CUDA_CHECK(cudaFree(dirs));

    return 0;
}

__global__ void pointGenKernel(float* points, float* dirs, int nelems,
                               float minimum, float step) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < nelems && j < nelems) {
        int id = i + j * nelems;

        float x = minimum + i * step;
        float y = minimum + j * step;

        points[2 * id + 0] = x;
        points[2 * id + 1] = y;

        dirs[2 * id + 0] = sinf(2.0f * PI * x / 10.f);
        dirs[2 * id + 1] = sinf(2.0f * PI * y / 10.f);
    }
}

void generatePoints(float* points, float* dirs) {
    static dim3 threads(8, 8);
    dim3 blocks(divup((int)(NELEMS), threads.x),
                divup((int)(NELEMS), threads.y));

    // clang-format off
    pointGenKernel<<<blocks, threads>>>(points, dirs, (int)(NELEMS), MINIMUM,
                                        STEP);
    // clang-format on
}
