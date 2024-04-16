add_library(theia_c_api_obj_lib OBJECT
    ${CMAKE_CURRENT_LIST_DIR}/chart.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exception.cpp
    ${CMAKE_CURRENT_LIST_DIR}/font.cpp
    ${CMAKE_CURRENT_LIST_DIR}/histogram.cpp
    ${CMAKE_CURRENT_LIST_DIR}/image.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plot.cpp
    ${CMAKE_CURRENT_LIST_DIR}/surface.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vector_field.cpp
    ${CMAKE_CURRENT_LIST_DIR}/window.cpp
    )

fg_set_target_compilation_props(theia_c_api_obj_lib)

target_include_directories(theia_c_api_obj_lib
    PRIVATE
    ${theia_SOURCE_DIR}/src/backend/opengl # For backend native headers
    $<TARGET_PROPERTY:theia_wtk,INTERFACE_INCLUDE_DIRECTORIES>
    )
