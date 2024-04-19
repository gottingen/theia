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

#include <theia/fg/image.h>
#include <theia/fg/window.h>

#include <error.hpp>

#include <utility>

namespace theia {
Image::Image(const unsigned pWidth, const unsigned pHeight,
             const ChannelFormat pFormat, const dtype pDataType)
    : mValue(0) {
    fg_image temp = 0;
    FG_THROW(
        fg_create_image(&temp, pWidth, pHeight, pFormat, (fg_dtype)pDataType));

    std::swap(mValue, temp);
}

Image::Image(const Image& pOther) {
    fg_image temp = 0;

    FG_THROW(fg_retain_image(&temp, pOther.get()));

    std::swap(mValue, temp);
}

Image::Image(const fg_image pHandle) : mValue(pHandle) {}

Image::~Image() { fg_release_image(get()); }

void Image::setAlpha(const float pAlpha) {
    FG_THROW(fg_set_image_alpha(get(), pAlpha));
}

void Image::keepAspectRatio(const bool pKeep) {
    FG_THROW(fg_set_image_aspect_ratio(get(), pKeep));
}

unsigned Image::width() const {
    unsigned temp = 0;
    FG_THROW(fg_get_image_width(&temp, get()));
    return temp;
}

unsigned Image::height() const {
    unsigned temp = 0;
    FG_THROW(fg_get_image_height(&temp, get()));
    return temp;
}

ChannelFormat Image::pixelFormat() const {
    fg_channel_format retVal = (fg_channel_format)0;
    FG_THROW(fg_get_image_pixelformat(&retVal, get()));
    return retVal;
}

theia::dtype Image::channelType() const {
    fg_dtype temp = (fg_dtype)1;
    FG_THROW(fg_get_image_type(&temp, get()));
    return (theia::dtype)temp;
}

unsigned Image::pixels() const {
    unsigned retVal = 0;
    FG_THROW(fg_get_pixel_buffer(&retVal, get()));
    return retVal;
}

unsigned Image::size() const {
    unsigned retVal = 0;
    FG_THROW(fg_get_image_size(&retVal, get()));
    return retVal;
}

void Image::render(const Window& pWindow, const int pX, const int pY,
                   const int pVPW, const int pVPH) const {
    FG_THROW(fg_render_image(pWindow.get(), get(), pX, pY, pVPW, pVPH));
}

fg_image Image::get() const { return mValue; }
}  // namespace theia
