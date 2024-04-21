cmake_minimum_required(VERSION 3.16.3)

project(theia VERSION 1.1.8 LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMakeModules")
list(APPEND CMAKE_PREFIX_PATH "${theia_BINARY_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${theia_SOURCE_DIR}/carbin_cmake")
list(APPEND CMAKE_MODULE_PATH "${theia_SOURCE_DIR}/carbin_cmake/arch")
list(APPEND CMAKE_MODULE_PATH "${theia_SOURCE_DIR}/carbin_cmake/copts")
list(APPEND CMAKE_MODULE_PATH "${theia_SOURCE_DIR}/carbin_cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${theia_SOURCE_DIR}/carbin_cmake/package")
list(APPEND CMAKE_MODULE_PATH "${theia_SOURCE_DIR}/carbin_cmake/recipes")

include(theia_package_config)
include(TheiaBuildType)
include(TheiaInstallDirs)
include(TheiaInternalUtils)
include(TheiaVersion)

set_policies(TYPE NEW POLICIES CMP0072 CMP0077)

option(BUILD_SHARED_LIBS "Build shared/static library" ON)

find_package(Boost      REQUIRED)
find_package(FontConfig QUIET)
find_package(FreeImage  QUIET)
find_package(Freetype   REQUIRED)
find_package(Sphinx     QUIET)

if(UNIX)
    dependency_check(FontConfig_FOUND
        "FontConfig is required on non-windows OS")
endif()

option(FG_BUILD_EXAMPLES
    "Build Examples" OFF)
option(FG_WITH_FREEIMAGE
    "Use FreeImage to add support for saving framebuffer to disk"
    ${FreeImage_FOUND})

option(FG_USE_STATIC_FREEIMAGE
    "Use static version of freeimage" OFF)
option(FG_USE_STATIC_CPPFLAGS
    "Use static libstdc++ & libgcc for generating theia library" OFF)

set(FG_USE_WINDOW_TOOLKIT "glfw3" CACHE STRING "Choose Window toolkit")
set_property(CACHE FG_USE_WINDOW_TOOLKIT PROPERTY STRINGS "glfw3" "sdl2")

set(FG_RENDERING_BACKEND "OpenGL" CACHE STRING "Choose Rendering Backend")
set_property(CACHE FG_RENDERING_BACKEND PROPERTY STRINGS "OpenGL")

mark_as_advanced(
    FG_USE_STATIC_FREEIMAGE
    FG_USE_STATIC_CPPFLAGS)

fg_deprecate(BUILD_DOCS FG_BUILD_DOCS)
fg_deprecate(BUILD_EXAMPLES FG_BUILD_EXAMPLES)
fg_deprecate(WITH_FREEIMAGE FG_WITH_FREEIMAGE)
fg_deprecate(USE_STATIC_FREEIMAGE FG_USE_STATIC_FREEIMAGE)
fg_deprecate(WITH_TOOLKIT FG_USE_WINDOW_TOOLKIT)

if(Boost_FOUND AND NOT TARGET Boost::boost)
    add_library(Boost::boost INTERFACE IMPORTED)
    set_property(TARGET Boost::boost
        PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif(Boost_FOUND AND NOT TARGET Boost::boost)


add_subdirectory(src/glad)
add_subdirectory(src/backend/common)
add_subdirectory(src/backend/glsl_shaders)
add_subdirectory(src/api/c)
add_subdirectory(src/api/cpp)
add_subdirectory(src/backend)

#--------------------------------------------------------------------
# Install include folder, docs, examples etc.
#--------------------------------------------------------------------
install(DIRECTORY include/
    DESTINATION ${FG_INSTALL_INC_DIR}
    COMPONENT theia_dev
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".gitignore" EXCLUDE)

install(FILES ${theia_BINARY_DIR}/include/theia/fg/version.h
    DESTINATION "${FG_INSTALL_INC_DIR}/theia/fg/"
    COMPONENT theia_dev)

# install the examples irrespective of the FG_BUILD_EXAMPLES value
# only the examples source files are installed, so the installation of these
# source files does not depend on FG_BUILD_EXAMPLES
# when FG_BUILD_EXAMPLES is OFF, the examples source is installed without
# building the example executables
install(DIRECTORY examples/ #NOTE The slash at the end is important
    DESTINATION ${FG_INSTALL_EXAMPLE_DIR}
    COMPONENT theia_dev)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${theia_BINARY_DIR}/theiaConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)


# export install config file
set(INCLUDE_DIRS include)
set(CMAKE_DIR ${FG_INSTALL_CMAKE_DIR})
configure_package_config_file(
    "${theia_SOURCE_DIR}/CMakeModules/theiaConfig.cmake.in"
    "cmake_install/theiaConfig.cmake"
    INSTALL_DESTINATION "${FG_INSTALL_CMAKE_DIR}"
    PATH_VARS INCLUDE_DIRS CMAKE_DIR
)
install(FILES ${theia_BINARY_DIR}/cmake_install/theiaConfig.cmake
    ${theia_BINARY_DIR}/theiaConfigVersion.cmake
    DESTINATION ${FG_INSTALL_CMAKE_DIR}
    COMPONENT theia_dev
)
install(EXPORT theiaTargets
    NAMESPACE theia::
    DESTINATION ${FG_INSTALL_CMAKE_DIR}
    COMPONENT theia_dev
)

# export build tree targets config file
set(INCLUDE_DIRS "${theia_SOURCE_DIR}/include" "${theia_BINARY_DIR}/include")
set(CMAKE_DIR "${theia_BINARY_DIR}")
configure_package_config_file(
    "${theia_SOURCE_DIR}/CMakeModules/theiaConfig.cmake.in"
    "theiaConfig.cmake"
    INSTALL_DESTINATION "${theia_BINARY_DIR}"
    PATH_VARS INCLUDE_DIRS CMAKE_DIR
    INSTALL_PREFIX "${theia_BINARY_DIR}"
)
export(EXPORT theiaTargets
    NAMESPACE theia::
    FILE theiaTargets.cmake
)

conditional_directory(FG_BUILD_EXAMPLES examples)

include(carbin_cpack)