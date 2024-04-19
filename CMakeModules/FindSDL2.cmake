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

#
# Finds the fontconfig libraries
#
# Sets the following variables:
#          SDL2_FOUND
#          SDL2_INCLUDE_DIR
#          SDL2_LIBRARY
#
# To help locate the library and include file, you could define an environment variable called
# SDL2_ROOT which points to the root of the glfw library installation. This is pretty useful
# on a Windows platform.
#
# Usage:
# find_package(SDL2)
# if (SDL2_FOUND)
#    target_link_libraries(mylib PRIVATE SDL2::SDL2)
# endif (SDL2_FOUND)
#
# NOTE: You do not need to include the SDL2 include directories since they
# will be included as part of the target_link_libraries command

find_path(SDL2_INCLUDE_DIR
    NAMES SDL.h
    PATHS
        ENV SDL2_ROOT
    PATH_SUFFIXES
        SDL2
        include/SDL2
        include
    DOC "Path to SDL2 include directory."
)

find_library(SDL2_LIBRARY
    NAMES SDL2
    PATHS
        ENV SDL2_ROOT
        /usr
        /usr/local
        /usr/lib/x86_64-linux-gnu
        /sw
        /opt/local
        $ENV{SDL2_ROOT}/lib-msvc110
        $ENV{SDL2_ROOT}/lib-msvc120
    PATH_SUFFIXES
        lib
        lib64
        lib/x64
        release
        debug
    DOC "Absolute path to SDL2 library."
    )

if (DEFINED SDL2_INCLUDE_DIR-NOTFOUND OR
        DEFINED SDL2_LIBRARY-NOTFOUND)
    message(SEND_ERROR "SDL2 not found")
endif ()

mark_as_advanced(
    SDL2_INCLUDE_DIR
    SDL2_LIBRARY
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL2
    REQUIRED_VARS SDL2_LIBRARY SDL2_INCLUDE_DIR
    )

if (SDL2_FOUND AND NOT TARGET SDL2::SDL2)
    add_library(SDL2::SDL2 UNKNOWN IMPORTED)
    set_target_properties(SDL2::SDL2 PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGAE "C"
        IMPORTED_LOCATION ${SDL2_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${SDL2_INCLUDE_DIR})
endif ()
