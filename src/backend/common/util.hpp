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

// This file contains platform agnostic utility functions

#pragma once

#define GLM_FORCE_RADIANS
#include <theia/glm/glm.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace theia {
namespace common {

/* clamp the float to [0-1] range
 *
 * @pValue is the value to be clamped
 */
float clampTo01(const float pValue);

#if defined(OS_WIN)
/* Get the paths to font files in Windows system directory
 *
 * @pFiles is the output vector to which font file paths are appended to.
 * @pDir is the directory from which font files are looked up
 * @pExt is the target font file extension we are looking for.
 */
void getFontFilePaths(std::vector<std::string>& pFiles, const std::string& pDir,
                      const std::string& pExt);
#endif

std::string clipPath(std::string path, std::string str);

std::string getEnvVar(const std::string& key);

/* Convert float value to string with given precision
 *
 * @pVal is the float value whose string representation is requested.
 * @pFormat is the c-style printf format for floats
 *
 * @return is the string representation of input float value.
 */
std::string toString(const float pVal, const std::string pFormat);

/* Print glm::mat4 to std::cout stream */
std::ostream& operator<<(std::ostream&, const glm::mat4&);

/* Calculate rotation axis and amount of rotation of Arc Ball
 *
 * This computation requires previous and current mouse cursor positions
 * which are the input parameters to this function call
 *
 * @lastPos previous mouse position
 * @currPos current mouse position
 *
 * @return Rotation axis vector and the angle of rotation
 * */
std::pair<glm::vec3, float> calcRotationFromArcBall(const glm::vec2& lastPos,
                                                    const glm::vec2& currPos,
                                                    const glm::vec4& viewport);

}  // namespace common
}  // namespace theia
