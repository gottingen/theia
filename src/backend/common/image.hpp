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

#include <backend.hpp>
#include <theia/glm/glm.hpp>
#include <image_impl.hpp>

#include <cstdint>
#include <memory>

namespace theia {
namespace common {

class Image {
   private:
    std::shared_ptr<detail::image_impl> mImage;

   public:
    Image(const unsigned pWidth, const unsigned pHeight,
          const theia::ChannelFormat pFormat, const theia::dtype pDataType)
        : mImage(std::make_shared<detail::image_impl>(pWidth, pHeight, pFormat,
                                                      pDataType)) {}

    Image(const fg_image pOther) {
        mImage = reinterpret_cast<Image *>(pOther)->impl();
    }

    inline const std::shared_ptr<detail::image_impl> &impl() const {
        return mImage;
    }

    inline void setAlpha(const float pAlpha) { mImage->setAlpha(pAlpha); }

    inline void keepAspectRatio(const bool pKeep) {
        mImage->keepAspectRatio(pKeep);
    }

    inline unsigned width() const { return mImage->width(); }

    inline unsigned height() const { return mImage->height(); }

    inline theia::ChannelFormat pixelFormat() const {
        return mImage->pixelFormat();
    }

    inline theia::dtype channelType() const { return mImage->channelType(); }

    inline unsigned pbo() const { return mImage->pbo(); }

    inline uint32_t size() const { return mImage->size(); }

    inline void render(const int pWindowId, const int pX, const int pY,
                       const int pVPW, const int pVPH, const glm::mat4 &pView,
                       const glm::mat4 &pOrient) const {
        mImage->render(pWindowId, pX, pY, pVPW, pVPH, pView, pOrient);
    }
};

}  // namespace common
}  // namespace theia
