
add_library(glad_obj_lib OBJECT glad.c)

set_target_properties(glad_obj_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(glad_obj_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

