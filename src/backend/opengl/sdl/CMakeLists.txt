find_package(SDL2)
mark_as_advanced(SDL2_DIR)

add_library(theia_wtk OBJECT
    ${CMAKE_CURRENT_LIST_DIR}/window.hpp
    ${CMAKE_CURRENT_LIST_DIR}/window.cpp
    )

fg_set_target_compilation_props(theia_wtk)

target_include_directories(theia_wtk
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    PRIVATE
    ${theia_SOURCE_DIR}/src/backend/opengl # backend's native headers
    )
