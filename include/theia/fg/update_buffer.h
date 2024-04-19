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

#pragma once

#include <theia/fg/defines.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup util_functions
 * @{
 */

/**
    Update backend specific vertex buffer from given host side memory

    \param[in] pBufferId is the buffer identifier
    \param[in] pBufferSize is the buffer size in bytes
    \param[in] pBufferData is the pointer of the host side memory

    \return \ref fg_err error code
 */
FGAPI fg_err fg_update_vertex_buffer(const unsigned pBufferId,
                                     const size_t pBufferSize,
                                     const void* pBufferData);

/**
    Update backend specific pixel buffer from given host side memory

    \param[in] pBufferId is the buffer identifier
    \param[in] pBufferSize is the buffer size in bytes
    \param[in] pBufferData is the pointer of the host side memory

    \return \ref fg_err error code
 */
FGAPI fg_err fg_update_pixel_buffer(const unsigned pBufferId,
                                    const size_t pBufferSize,
                                    const void* pBufferData);

/**
    Sync all rendering operations till this point

    \return \ref fg_err error code
 */
FGAPI fg_err fg_finish();

/** @} */

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace theia
{

/**
    Update backend specific vertex buffer from given host side memory

    \param[in] pBufferId is the buffer identifier
    \param[in] pBufferSize is the buffer size in bytes
    \param[in] pBufferData is the pointer of the host side memory

    \ingroup util_functions
 */
FGAPI void updateVertexBuffer(const unsigned pBufferId,
                              const size_t pBufferSize,
                              const void* pBufferData);

/**
    Update backend specific pixel buffer from given host side memory

    \param[in] pBufferId is the buffer identifier
    \param[in] pBufferSize is the buffer size in bytes
    \param[in] pBufferData is the pointer of the host side memory

    \ingroup util_functions
 */
FGAPI void updatePixelBuffer(const unsigned pBufferId,
                             const size_t pBufferSize,
                             const void* pBufferData);

/**
    Sync all rendering operations till this point

    \ingroup util_functions
 */
FGAPI void finish();

}
#endif
