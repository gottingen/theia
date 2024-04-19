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

#include <window.hpp>

#include <chart_impl.hpp>
#include <colormap_impl.hpp>
#include <common/defines.hpp>
#include <font_impl.hpp>
#include <image_impl.hpp>
#include <plot_impl.hpp>

#include <memory>

namespace theia {
namespace opengl {

class window_impl {
   private:
    long long mCxt;
    long long mDsp;
    int mID;
    std::unique_ptr<wtk::Widget> mWidget;

    std::shared_ptr<font_impl> mFont;
    std::shared_ptr<colormap_impl> mCMap;
    std::shared_ptr<plot_impl> mArcBallLoop0;
    std::shared_ptr<plot_impl> mArcBallLoop1;

    uint32_t mColorMapUBO;
    uint32_t mUBOSize;

    void prepArcBallObjects();

   public:
    window_impl(int pWidth, int pHeight, const char* pTitle,
                std::weak_ptr<window_impl> pWindow,
                const bool invisible = false);

    ~window_impl();

    void makeContextCurrent();
    void setFont(const std::shared_ptr<font_impl>& pFont);
    void setTitle(const char* pTitle);
    void setPos(int pX, int pY);
    void setSize(unsigned pWidth, unsigned pHeight);
    void setColorMap(theia::ColorMap cmap);

    int getID() const;
    long long context() const;
    long long display() const;
    int width() const;
    int height() const;
    const std::unique_ptr<wtk::Widget>& get() const;
    const std::shared_ptr<colormap_impl>& colorMapPtr() const;

    void hide();
    void show();
    bool close();

    void draw(const std::shared_ptr<AbstractRenderable>& pRenderable);

    void draw(const int pRows, const int pCols, const int pIndex,
              const std::shared_ptr<AbstractRenderable>& pRenderable,
              const char* pTitle);

    void swapBuffers();

    void saveFrameBuffer(const char* pFullPath);
};

}  // namespace opengl
}  // namespace theia
