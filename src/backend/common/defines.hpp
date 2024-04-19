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

#include <boost/functional/hash.hpp>
#include <common/util.hpp>

#include <unordered_map>
#include <utility>

namespace theia {
namespace common {

using CellIndex     = std::tuple<int, int, int>;
using MatrixHashMap = std::unordered_map<CellIndex, glm::mat4>;

constexpr int ARCBALL_CIRCLE_POINTS = 100;
constexpr float MOVE_SPEED          = 0.005f;
constexpr float ZOOM_SPEED          = 0.0075f;
constexpr float EPSILON             = 1.0e-6f;
constexpr float ARC_BALL_RADIUS     = 0.75f;
constexpr double PI                 = 3.14159265358979323846;
constexpr float BLACK[]             = {0.0f, 0.0f, 0.0f, 1.0f};
constexpr float GRAY[]              = {0.75f, 0.75f, 0.75f, 1.0f};
constexpr float WHITE[]             = {1.0f, 1.0f, 1.0f, 1.0f};
constexpr float AF_BLUE[]           = {0.0588f, 0.1137f, 0.2745f, 1.0f};
static const glm::mat4 IDENTITY(1.0f);

#if defined(OS_WIN)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#define __FG_FILENAME__ (theia::common::clipPath(__FILE__, "src\\").c_str())
#else
#define __FG_FILENAME__ (theia::common::clipPath(__FILE__, "src/").c_str())
#endif

}  // namespace common
}  // namespace theia

namespace std {

template<>
struct hash<theia::common::CellIndex> {
    std::size_t operator()(const theia::common::CellIndex& key) const {
        size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(key));
        boost::hash_combine(seed, std::get<1>(key));
        boost::hash_combine(seed, std::get<2>(key));
        return seed;
    }
};

}  // namespace std
