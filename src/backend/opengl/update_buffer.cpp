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

#include <common/err_handling.hpp>
#include <theia/fg/update_buffer.h>
#include <gl_helpers.hpp>

fg_err fg_update_vertex_buffer(const unsigned pBufferId,
                               const size_t pBufferSize,
                               const void* pBufferData) {
    try {
        glBindBuffer(GL_ARRAY_BUFFER, pBufferId);
        glBufferSubData(GL_ARRAY_BUFFER, 0, pBufferSize, pBufferData);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    CATCHALL;

    return FG_ERR_NONE;
}

fg_err fg_update_pixel_buffer(const unsigned pBufferId,
                              const size_t pBufferSize,
                              const void* pBufferData) {
    try {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pBufferId);
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, pBufferSize, pBufferData);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    CATCHALL;

    return FG_ERR_NONE;
}

fg_err fg_finish() {
    try {
        glFinish();
    }
    CATCHALL;

    return FG_ERR_NONE;
}

namespace theia {
void updateVertexBuffer(const unsigned pBufferId, const size_t pBufferSize,
                        const void* pBufferData) {
    fg_err val = fg_update_vertex_buffer(pBufferId, pBufferSize, pBufferData);
    if (val != FG_ERR_NONE) FG_ERROR("Vertex Buffer Object update failed", val);
}

void updatePixelBuffer(const unsigned pBufferId, const size_t pBufferSize,
                       const void* pBufferData) {
    fg_err val = fg_update_pixel_buffer(pBufferId, pBufferSize, pBufferData);
    if (val != FG_ERR_NONE) FG_ERROR("Pixel Buffer Object update failed", val);
}

void finish() {
    fg_err val = fg_finish();
    if (val != FG_ERR_NONE) FG_ERROR("glFinish failed", val);
}
}  // namespace theia
