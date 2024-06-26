cmake_minimum_required(VERSION 3.16.3)

project(theia-Examples LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

set(THEIA_BUILD_TREE OFF)
if(theia_BINARY_DIR AND EXISTS "${theia_BINARY_DIR}")
  set(theia_DIR ${theia_BINARY_DIR})
  set(THEIA_BUILD_TREE ON)
endif()

find_package(CUDA   QUIET)
find_package(theia  REQUIRED)

include(CMakeParseArguments)
include(CMakeDependentOption)
include(InternalUtils)

cmake_dependent_option(FG_BUILD_CUDA_EXAMPLES
  "Build cuda examples" ON
  "CUDA_FOUND;FG_BUILD_EXAMPLES" OFF)

fg_deprecate(BUILD_EXAMPLES_CUDA FG_BUILD_CUDA_EXAMPLES)

function(add_example target_name source backend)
    set(options CXX11)
    set(single_value_args "")
    set(multiple_value_args INCLUDE_DIRS LIBRARIES)
    cmake_parse_arguments(arg "${options}" "${single_value_args}" "${multiple_value_args}" ${ARGN})

    set(target "example_${target_name}_${backend}")

    string(TOLOWER ${backend} lowerCaseBackend)

    if (${lowerCaseBackend} STREQUAL "cuda")
        set(native_cc_flags "")
        if (WIN32)
            set(native_cc_flags
                "-Xcompiler /wd4275 -Xcompiler /bigobj -Xcompiler /EHsc")
        endif ()
        if (THEIA_BUILD_TREE)
          CUDA_INCLUDE_DIRECTORIES(
            ${theia_BINARY_DIR}/include
            ${theia_SOURCE_DIR}/include
          )
        else ()
          get_target_property(
            THEIA_INC_DIRS theia::theia INTERFACE_INCLUDE_DIRECTORIES)
          CUDA_INCLUDE_DIRECTORIES(${THEIA_INC_DIRS})
        endif ()
        cuda_add_executable(${target} ${source}
          OPTIONS
            "${native_cc_flags} -D_FORCE_INLINES -Xcudafe \"--diag_suppress=1388\""
        )
    else ()
        add_executable(${target} ${source})
    endif ()

    set_target_properties(${target} PROPERTIES OUTPUT_NAME "${target_name}")
    set_target_properties(${target} PROPERTIES FOLDER Examples/${backend})

    if (${arg_CXX11})
        set_target_properties(${target} PROPERTIES CXX_STANDARD 11)
    endif (${arg_CXX11})

    if(WIN32)
        target_compile_definitions(${target} PRIVATE OS_WIN WIN32_MEAN_AND_LEAN)
    elseif(APPLE)
        target_compile_definitions(${target} PRIVATE OS_MAC)
    else(WIN32)
        target_compile_definitions(${target} PRIVATE OS_LNX)
    endif(WIN32)

    target_include_directories(${target} PRIVATE ${arg_INCLUDE_DIRS})
    if(${lowerCaseBackend} STREQUAL "cuda")
        # CUDA backend examples need same LINK keyword(PRIVATE|PUBLIC) used for
        # example targets i.e. used by FindCUDA which is empty by default
        target_link_libraries(${target} theia::theia ${arg_LIBRARIES})
    else()
        target_link_libraries(${target} PRIVATE theia::theia ${arg_LIBRARIES})
    endif()
endfunction()

add_subdirectory(cpu)
conditional_directory(FG_BUILD_CUDA_EXAMPLES cuda)

