# BkndTargetName is the target used by
# rendering backend(opengl/vulkan) to add
# source files and set target properties


# Until we move up minimum cmake version to 3.14
# add_library and target_link_libraryies have to be
# at same level or one level above in cmakelists-file-tree
add_library(theia "")

fg_set_target_compilation_props(theia)
set_target_properties(theia
    PROPERTIES
    VERSION ${theia_VERSION}
    SOVERSION ${theia_VERSION_MAJOR}
    )

if(${FG_RENDERING_BACKEND} STREQUAL "OpenGL")
    find_package(OpenGL REQUIRED)
    add_subdirectory(opengl)
    target_link_libraries(theia PUBLIC OpenGL::GL)
    if(${FG_USE_WINDOW_TOOLKIT} STREQUAL "glfw3")
        include(opengl/glfw/CMakeLists.txt)
        target_link_libraries(theia PRIVATE glfw)
    elseif(${FG_USE_WINDOW_TOOLKIT} STREQUAL "sdl2")
        include(opengl/sdl/CMakeLists.txt)
        target_link_libraries(theia PRIVATE SDL2::SDL2)
    endif()
elseif(${FG_RENDERING_BACKEND} STREQUAL "Vulkan") # vulkan arm is not possible now
    message(FATAL_ERROR "Vulkan rendering support not added yet")
endif()

add_dependencies(theia ${glsl_shader_targets})

target_sources(theia
    PRIVATE
    $<TARGET_OBJECTS:theia_wtk>
    $<TARGET_OBJECTS:theia_common_obj_lib>
    $<TARGET_OBJECTS:theia_c_api_obj_lib>
    $<TARGET_OBJECTS:theia_cpp_api_obj_lib>
    )

target_include_directories(theia SYSTEM PRIVATE ${FREETYPE_INCLUDE_DIRS})
target_include_directories(theia
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    $<TARGET_PROPERTY:theia_wtk,INTERFACE_INCLUDE_DIRECTORIES>
    )

if(FG_WITH_FREEIMAGE)
    target_compile_definitions(theia PRIVATE USE_FREEIMAGE)
    if(FG_USE_STATIC_FREEIMAGE)
        target_link_libraries(theia PUBLIC FreeImage::FreeImage_STATIC)
    else()
        target_link_libraries(theia PRIVATE FreeImage::FreeImage)
    endif()
endif()

target_link_libraries(theia
    PRIVATE
    Boost::boost
    ${FREETYPE_LIBRARIES}
    ${CMAKE_DL_LIBS}
    )

target_link_libraries(theia PRIVATE glad)

if(UNIX)
    target_link_libraries(theia
        PRIVATE
        FontConfig::FontConfig
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<BOOL:${FG_USE_STATIC_CPPFLAGS}>>:-static-libstdc++>
        $<$<AND:$<C_COMPILER_ID:GNU>,$<BOOL:${FG_USE_STATIC_CPPFLAGS}>>:-static-libgcc>
        )
endif(UNIX)

source_group(include
    REGULAR_EXPRESSION ${theia_SOURCE_DIR}/include/*)
source_group("api\\cpp"
    REGULAR_EXPRESSION ${theia_SOURCE_DIR}/src/api/cpp/*)
source_group("api\\c"
    REGULAR_EXPRESSION ${theia_SOURCE_DIR}/src/api/c/*)
source_group("backend"
    REGULAR_EXPRESSION ${theia_SOURCE_DIR}/src/backend/common/*)
source_group("backend\\shaders"
    REGULAR_EXPRESSION ${theia_SOURCE_DIR}/src/backend/glsl_shaders/*)
source_group("backend"
    REGULAR_EXPRESSION ${CMAKE_CURRENT_LIST_DIR}/*)
if(${FG_RENDERING_BACKEND} STREQUAL "OpenGL")
    if(${FG_USE_WINDOW_TOOLKIT} STREQUAL "glfw3")
        source_group("backend\\glfw"
            REGULAR_EXPRESSION ${theia_SOURCE_DIR}/src/backend/opengl/glfw/*)
    elseif(${FG_USE_WINDOW_TOOLKIT} STREQUAL "sdl2")
        source_group("backend\\sdl2"
            REGULAR_EXPRESSION ${theia_SOURCE_DIR}/src/backend/opengl/sdl/*)
    endif()
elseif(${FG_RENDERING_BACKEND} STREQUAL "Vulkan") # vulkan arm is not possible now
    message(FATAL_ERROR "Vulkan rendering support not added yet")
endif()

#--------------------------------------------------------------------
# Install theia target
#--------------------------------------------------------------------
install(TARGETS theia
    EXPORT theiaTargets
    COMPONENT theia
    PUBLIC_HEADER DESTINATION theia/fg
    RUNTIME DESTINATION ${FG_INSTALL_BIN_DIR}
    LIBRARY DESTINATION ${FG_INSTALL_LIB_DIR}
    ARCHIVE DESTINATION ${FG_INSTALL_LIB_DIR}
    FRAMEWORK DESTINATION framework
    INCLUDES DESTINATION ${FG_INSTALL_INC_DIR})

# install dependencies
if(WIN32 AND BUILD_SHARED_LIBS)
    if(FG_WITH_FREEIMAGE AND NOT FG_USE_STATIC_FREEIMAGE)
        install(FILES $<TARGET_FILE:FreeImage::FreeImage>
            DESTINATION ${FG_INSTALL_BIN_DIR}
            COMPONENT theia_dependencies)
    endif()
    if(${FG_USE_WINDOW_TOOLKIT} STREQUAL "glfw3")
        install(FILES $<TARGET_FILE:glfw>
            DESTINATION ${FG_INSTALL_BIN_DIR}
            COMPONENT theia_dependencies)
    elseif(${FG_USE_WINDOW_TOOLKIT} STREQUAL "sdl2")
        install(FILES $<TARGET_FILE:SDL2::SDL2>
            DESTINATION ${FG_INSTALL_BIN_DIR}
            COMPONENT theia_dependencies)
    endif()
endif()

