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

const unsigned DIMX   = 512;
const unsigned DIMY   = 512;
const size_t TOT_SIZE = DIMX * DIMY * 4;

void kernel(unsigned char* dev_out);

int main(void) {
    unsigned char* dev_out;

    /*
     * First theia call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other theia::* object to be created successfully
     */
    theia::Window wnd(DIMX, DIMY, "Fractal Demo");
    wnd.makeCurrent();

    /* Create an image object which creates the necessary
     * textures and pixel buffer objects to hold the image
     * */
    theia::Image img(DIMX, DIMY, FG_RGBA, theia::u8);

    GfxHandle* handle = 0;

    // create GL-CPU interop buffer
    createGLBuffer(&handle, img.pixels(), THEIA_IMAGE_BUFFER);

    /* copy your data into the pixel buffer object exposed by
     * theia::Image class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, theia provides copy headers
     * along with the library to help with this task
     */
    THEIA_CUDA_CHECK(cudaMalloc((void**)&dev_out, TOT_SIZE));
    kernel(dev_out);

    // copy the data from compute buffer to graphics buffer
    copyToGLBuffer(handle, (ComputeResourceHandle)dev_out, img.size());

    do { wnd.draw(img); } while (!wnd.close());

    // destroy GL-CPU Interop buffer
    releaseGLBuffer(handle);
    THEIA_CUDA_CHECK(cudaFree(dev_out));
    return 0;
}

__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx          = scale * (float)(DIMX / 2.0f - x) / (DIMX / 2.0f);
    float jy          = scale * (float)(DIMY / 2.0f - y) / (DIMY / 2.0f);

    cuFloatComplex c = make_cuFloatComplex(-0.8f, 0.156f);
    cuFloatComplex a = make_cuFloatComplex(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = cuCaddf(cuCmulf(a, a), c);
        if (cuCabsf(a) > 1000.0f) return 0;
    }

    return 1;
}

__global__ void julia(unsigned char* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < DIMX && y < DIMY) {
        int offset = x + y * DIMX;
        // now calculate the value at that position
        int juliaValue = julia(x, y);

        out[offset * 4 + 2] = 255 * juliaValue;
        out[offset * 4 + 0] = 0;
        out[offset * 4 + 1] = 0;
        out[offset * 4 + 3] = 255;
    }
}

inline int divup(int a, int b) { return (a + b - 1) / b; }

void kernel(unsigned char* dev_out) {
    static const dim3 threads(8, 8);
    dim3 blocks(divup(DIMX, threads.x), divup(DIMY, threads.y));

    // clang-format off
    julia<<<blocks, threads>>>(dev_out);
    // clang-format on
}
