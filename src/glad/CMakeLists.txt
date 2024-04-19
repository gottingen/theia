#
# Copyright 2023 The EA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

add_library(glad STATIC glad.c)
set_target_properties(glad PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(glad PRIVATE ${theia_SOURCE_DIR}/include)

install(TARGETS glad
        EXPORT theiaTargets
        COMPONENT glad
        PUBLIC_HEADER DESTINATION theia/glad
        RUNTIME DESTINATION ${FG_INSTALL_BIN_DIR}
        LIBRARY DESTINATION ${FG_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${FG_INSTALL_LIB_DIR}
        FRAMEWORK DESTINATION framework
        INCLUDES DESTINATION ${FG_INSTALL_INC_DIR})