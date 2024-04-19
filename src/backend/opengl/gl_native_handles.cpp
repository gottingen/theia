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

#include <gl_native_handles.hpp>

#if defined(OS_WIN)
#include <windows.h>
#elif defined(OS_MAC)
#include <OpenGL/OpenGL.h>
#else
#include <GL/glx.h>
#endif

namespace theia {
namespace opengl {

ContextHandle getCurrentContextHandle() {
    auto id = ContextHandle{0};

#if defined(OS_WIN)
    const auto context = wglGetCurrentContext();
#elif defined(OS_LNX)
    const auto context = glXGetCurrentContext();
#else
    const auto context = CGLGetCurrentContext();
#endif
    id = reinterpret_cast<ContextHandle>(context);

    return id;
}

DisplayHandle getCurrentDisplayHandle() {
    auto id = DisplayHandle{0};

#if defined(OS_WIN)
    const auto display = wglGetCurrentDC();
#elif defined(OS_LNX)
    const auto display = glXGetCurrentDisplay();
#else
    const DisplayHandle display = 0;
#endif
    id = reinterpret_cast<DisplayHandle>(display);

    return id;
}

}  // namespace opengl
}  // namespace theia
