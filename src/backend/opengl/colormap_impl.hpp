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

#include <array>
#include <cstdint>

namespace theia {
namespace opengl {

constexpr unsigned int TheiaNumColorMaps = 11;

class colormap_impl {
   private:
    /*
     * READ THIS BEFORE ADDING NEW COLORMAP
     *
     * each of the following buffers will point
     * to the data from floating point arrays
     * defined in cmap.hpp header. Currently,
     * the largest colormap is 259 colors(1036 floats).
     * Hence the shader of internal::image_impl uses
     * uniform array of vec4 with size 259.
     * when a new colormap is added, make sure
     * the size of array declared in the shaders
     * used by *_impl objects to reflect appropriate
     * size */
    std::array<uint32_t, TheiaNumColorMaps> mMapIds;
    std::array<uint32_t, TheiaNumColorMaps> mMapLens;

   public:
    colormap_impl();
    ~colormap_impl();

    uint32_t cmapUniformBufferId(theia::ColorMap cmap) const;
    uint32_t cmapLength(theia::ColorMap cmap) const;
};

}  // namespace opengl
}  // namespace theia
