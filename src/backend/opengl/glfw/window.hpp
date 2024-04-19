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

#include <common/defines.hpp>
#include <gl_helpers.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <theia/glm/glm.hpp>

#include <memory>

/* the short form wtk stands for
 * Windowing Tool Kit */
namespace theia {
namespace wtk {

void initWindowToolkit();
void destroyWindowToolkit();

class Widget {
   private:
    GLFWwindow* mWindow;
    GLFWcursor* mRotationCursor;
    GLFWcursor* mZoomCursor;
    bool mClose;
    glm::vec2 mLastPos;
    int mButton;
    int mButtonAction;
    bool mRotationFlag;

    theia::common::MatrixHashMap mViewMatrices;
    theia::common::MatrixHashMap mOrientMatrices;

    Widget();

    const glm::vec4 getCellViewport(const glm::vec2& pos);
    const glm::mat4 findTransform(const theia::common::MatrixHashMap& pMap,
                                  const double pX, const double pY);
    const glm::mat4 getCellViewMatrix(const double pXPos, const double pYPos);
    const glm::mat4 getCellOrientationMatrix(const double pXPos,
                                             const double pYPos);
    void setTransform(theia::common::MatrixHashMap& pMap, const double pX,
                      const double pY, const glm::mat4& pMat);
    void setCellViewMatrix(const double pXPos, const double pYPos,
                           const glm::mat4& pMatrix);
    void setCellOrientationMatrix(const double pXPos, const double pYPos,
                                  const glm::mat4& pMatrix);

   public:
    /* public variables */
    int mWidth;   // Framebuffer width
    int mHeight;  // Framebuffer height

    /* Constructors and methods */
    Widget(int pWidth, int pHeight, const char* pTitle,
           const std::unique_ptr<Widget>& pWidget, const bool invisible);

    ~Widget();

    GLFWwindow* getNativeHandle() const;

    void makeContextCurrent() const;

    long long getGLContextHandle();

    long long getDisplayHandle();

    GLADloadproc getProcAddr();

    void setTitle(const char* pTitle);

    void setPos(int pX, int pY);

    void setSize(unsigned pW, unsigned pH);

    void swapBuffers();

    void hide();

    void show();

    bool close();

    void resetCloseFlag();

    void resizeHandler(int pWidth, int pHeight);

    void keyboardHandler(int pKey, int pScancode, int pAction, int pMods);

    void cursorHandler(double pXPos, double pYPos);

    void mouseButtonHandler(int pButton, int pAction, int pMods);

    void pollEvents();

    const glm::mat4 getViewMatrix(const theia::common::CellIndex& pIndex);
    const glm::mat4 getOrientationMatrix(
        const theia::common::CellIndex& pIndex);
    void resetViewMatrices();
    void resetOrientationMatrices();

    inline bool isBeingRotated() const { return mRotationFlag; }

    glm::vec2 getCursorPos() const;
};

}  // namespace wtk
}  // namespace theia
