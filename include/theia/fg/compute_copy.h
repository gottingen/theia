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

#ifndef __COMPUTE_DATA_COPY_H__
#define __COMPUTE_DATA_COPY_H__

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


#if defined(USE_THEIA_CPU_COPY_HELPERS)

// No special headers for cpu backend

#elif defined(USE_THEIA_CUDA_COPY_HELPERS)

#include <stdio.h>

#ifndef GL_VERSION
// gl.h is required by cuda_gl_interop to be included before it
// And gl.h requires windows.h to be included before it
#if defined(OS_WIN)
#include <windows.h>
#endif // OS_WIN
#include <GL/gl.h>
#endif // GL_VERSION

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#else

    #error "Invalid Compute model, exiting."

#endif


/// A backend-agnostic handle to a compute memory resource originating
/// from an OpenGL resource.
///
/// - cudaGraphicsResource in CUDA
/// - unsigned from standard cpu
#if defined(USE_THEIA_CPU_COPY_HELPERS)
/// OpenGL interop with CPU uses regular OpenGL buffer
typedef unsigned GfxResourceHandle;
#elif defined(USE_THEIA_CUDA_COPY_HELPERS)
/// OpenGL interop with CUDA uses an opaque CUDA object
typedef cudaGraphicsResource* GfxResourceHandle;
#endif


/** A backend-agnostic handle to a compute memory resource.

  For example:
    CUDA device pointer, like float*, int* from cudaMalloc.
  */
typedef void* ComputeResourceHandle;

/// Enum to indicate if OpenCL buffer is a PBO or VBO
typedef enum {
    THEIA_IMAGE_BUFFER  = 0,     ///< OpenGL Pixel Buffer Object
    THEIA_VERTEX_BUFFER = 1      ///< OpenGL Vertex Buffer Object
} BufferType;

/// A tuple object of GfxResourceHandle and \ref BufferType
typedef struct {
    GfxResourceHandle mId;
    BufferType mTarget;
} GfxHandle;


///////////////////////////////////////////////////////////////////////////////

#if defined(USE_THEIA_CPU_COPY_HELPERS)

static
void createGLBuffer(GfxHandle** pOut, const unsigned pResourceId, const BufferType pTarget)
{
    GfxHandle* temp = (GfxHandle*)malloc(sizeof(GfxHandle));

    temp->mId = pResourceId;
    temp->mTarget = pTarget;

    *pOut = temp;
}

static
void releaseGLBuffer(GfxHandle* pHandle)
{
    free(pHandle);
}

static
void copyToGLBuffer(GfxHandle* pGLDestination, ComputeResourceHandle  pSource, const size_t pSize)
{
    GfxHandle* temp = pGLDestination;

    if (temp->mTarget==THEIA_IMAGE_BUFFER) {
        fg_update_pixel_buffer(temp->mId, pSize, pSource);
    } else if (temp->mTarget==THEIA_VERTEX_BUFFER) {
        fg_update_vertex_buffer(temp->mId, pSize, pSource);
    }
}
#endif

///////////////////////////////////////////////////////////////////////////////

#if defined(USE_THEIA_CUDA_COPY_HELPERS)

static void handleCUDAError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define THEIA_CUDA_CHECK(err) (handleCUDAError(err, __FILE__, __LINE__ ))

static
void createGLBuffer(GfxHandle** pOut, const unsigned pResourceId, const BufferType pTarget)
{
    GfxHandle* temp = (GfxHandle*)malloc(sizeof(GfxHandle));

    temp->mTarget = pTarget;

    cudaGraphicsResource *cudaImageResource;

    THEIA_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaImageResource,
                                                  pResourceId,
                                                  cudaGraphicsMapFlagsWriteDiscard));

    temp->mId = cudaImageResource;

    *pOut = temp;
}

static
void releaseGLBuffer(GfxHandle* pHandle)
{
    THEIA_CUDA_CHECK(cudaGraphicsUnregisterResource(pHandle->mId));
    free(pHandle);
}

static
void copyToGLBuffer(GfxHandle* pGLDestination, ComputeResourceHandle  pSource, const size_t pSize)
{
    size_t numBytes;
    void* pointer = NULL;

    cudaGraphicsResource *cudaResource = pGLDestination->mId;

    THEIA_CUDA_CHECK(cudaGraphicsMapResources(1, &cudaResource, 0));

    THEIA_CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&pointer, &numBytes, cudaResource));

    THEIA_CUDA_CHECK(cudaMemcpy(pointer, pSource, numBytes, cudaMemcpyDeviceToDevice));

    THEIA_CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
}
#endif

#ifdef __cplusplus
}
#endif

#endif
