#
# Copyright 2023 The Carbin Authors.
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

include(CMakeParseArguments)
include(carbin_config_cxx_opts)
include(carbin_install_dirs)
include(carbin_print)

function(carbin_cc_binary)
    set(options
            PUBLIC
            EXCLUDE_SYSTEM
    )
    set(list_args
            DEPS
            SOURCES
            DEFINITIONS
            COPTS
            CXXOPTS
            CUOPTS
            LINKS
            INCLUDES
            )

    cmake_parse_arguments(
            CARBIN_CC_BINARY
            "${options}"
            "NAME"
            "${list_args}"
            ${ARGN}
    )

    set(${CARBIN_CC_BINARY_NAME}_INCLUDE_SYSTEM SYSTEM)
    if (CARBIN_CC_LIB_EXCLUDE_SYSTEM)
        set(${CARBIN_CC_BINARY_NAME}_INCLUDE_SYSTEM "")
    endif ()

    carbin_raw("-----------------------------------")
    carbin_print_label("Building Binary" "${CARBIN_CC_BINARY_NAME}")
    carbin_raw("-----------------------------------")
    if (VERBOSE_CARBIN_BUILD)
        carbin_print_list_label("Sources" CARBIN_CC_BINARY_SOURCES)
        carbin_print_list_label("Deps" CARBIN_CC_BINARY_DEPS)
        carbin_print_list_label("COPTS" CARBIN_CC_BINARY_COPTS)
        carbin_print_list_label("CXXOPTS" CARBIN_CC_BINARY_CXXOPTS)
        carbin_print_list_label("CUOPTS" CARBIN_CC_BINARY_CUOPTS)
        carbin_print_list_label("Defines" CARBIN_CC_BINARY_DEFINITIONS)
        carbin_print_list_label("Includes" CARBIN_CC_BINARY_INCLUDES)
        carbin_print_list_label("Links" CARBIN_CC_BINARY_LINKS)
        message("-----------------------------------")
    endif ()

    set(exec_case ${CARBIN_CC_BINARY_NAME})

    add_executable(${exec_case} ${CARBIN_CC_BINARY_SOURCES})

    target_compile_options(${exec_case} PRIVATE $<$<COMPILE_LANGUAGE:C>:${CARBIN_CC_BINARY_COPTS}>)
    target_compile_options(${exec_case} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CARBIN_CC_BINARY_CXXOPTS}>)
    target_compile_options(${exec_case} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CARBIN_CC_BINARY_CUOPTS}>)
    if (CARBIN_CC_BINARY_DEPS)
        add_dependencies(${exec_case} ${CARBIN_CC_BINARY_DEPS})
    endif ()
    target_link_libraries(${exec_case} PRIVATE ${CARBIN_CC_BINARY_LINKS})

    target_compile_definitions(${exec_case}
            PUBLIC
            ${CARBIN_CC_BINARY_DEFINITIONS}
            )

    target_include_directories(${exec_case} ${${CARBIN_CC_LIB_NAME}_INCLUDE_SYSTEM}
            PRIVATE
            ${CARBIN_CC_BINARY_INCLUDES}
            "$<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}>"
            "$<BUILD_INTERFACE:${${PROJECT_NAME}_BINARY_DIR}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
            )
    if (CARBIN_CC_BINARY_PUBLIC)
        install(TARGETS ${exec_case}
                EXPORT ${PROJECT_NAME}Targets
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                )
    endif (CARBIN_CC_BINARY_PUBLIC)

endfunction()
