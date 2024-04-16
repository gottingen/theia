
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