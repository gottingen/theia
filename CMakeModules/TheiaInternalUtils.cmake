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


function(dependency_check VAR ERROR_MESSAGE)
    if(NOT ${VAR})
        message(SEND_ERROR ${ERROR_MESSAGE})
    endif()
endfunction()

# Includes the directory if the variable is set
function(conditional_directory variable directory)
    if(${variable})
        add_subdirectory(${directory})
    endif()
endfunction()

function(get_native_path out_path path)
    file(TO_NATIVE_PATH ${path} native_path)
    string(REPLACE "\\" "\\\\" native_path  ${native_path})
    set(${out_path} ${native_path} PARENT_SCOPE)
endfunction()

function(__fg_deprecate_var var access value)
  if(access STREQUAL "READ_ACCESS")
      message(DEPRECATION "Variable ${var} is deprecated. Use FG_${var} instead.")
  endif()
endfunction()

function(fg_deprecate var newvar)
  if(DEFINED ${var})
    message(DEPRECATION "Variable ${var} is deprecated. Use ${newvar} instead.")
    get_property(doc CACHE ${newvar} PROPERTY HELPSTRING)
    set(${newvar} ${${var}} CACHE BOOL "${doc}" FORCE)
    unset(${var} CACHE)
  endif()
  variable_watch(${var} __fg_deprecate_var)
endfunction()

# mark CUDA cmake cache variables as advanced
# this should have been taken care of by FindCUDA I think.
mark_as_advanced(
    CMAKE_CUDA_HOST_COMPILER
    CUDA_HOST_COMPILER
    CUDA_SDK_ROOT_DIR
    CUDA_TOOLKIT_ROOT_DIR
    CUDA_USE_STATIC_CUDA_RUNTIME
    CUDA_rt_LIBRARY)

macro(set_policies)
    cmake_parse_arguments(SP "" "TYPE" "POLICIES" ${ARGN})
    foreach(_policy ${SP_POLICIES})
        if(POLICY ${_policy})
            cmake_policy(SET ${_policy} ${SP_TYPE})
        endif()
    endforeach()
endmacro()

# This function sets all common compilation flags, include directoriesm
# cmake target properties that are common to all OBJECT libraries and the
# finaly theia SHARED/STATIC library. Do not add any target specific options
# inside this function.
function(fg_set_target_compilation_props target)
    set_target_properties(${target}
        PROPERTIES
        FOLDER ${PROJECT_NAME}
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        CXX_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_INLINES_HIDDEN YES
        LINKER_LANGUAGE CXX
        )

    if(WIN32)
        target_compile_definitions(${target}
            PRIVATE FGDLL OS_WIN WIN32_MEAN_AND_LEAN)

        # C4068: Warnings about unknown pragmas
        # C4275: Warnings about using non-exported classes as base class of an
        #        exported class
        set_target_properties(${target} PROPERTIES COMPILE_FLAGS "/wd4068 /wd4275")
    elseif(APPLE)
        target_compile_definitions(${target} PRIVATE OS_MAC)
    else(WIN32)
        target_compile_definitions(${target} PRIVATE OS_LNX)
    endif(WIN32)

    target_include_directories(${target}
        SYSTEM PRIVATE
        $<TARGET_PROPERTY:Boost::boost,INTERFACE_INCLUDE_DIRECTORIES>
        )
    target_include_directories(${target}
        PUBLIC
        $<INSTALL_INTERFACE:${FG_INSTALL_INC_DIR}>
        $<BUILD_INTERFACE:${theia_SOURCE_DIR}/include> # build-tree public headers
        $<BUILD_INTERFACE:${theia_BINARY_DIR}/include> # build-tree generated headers
        PRIVATE
        ${theia_SOURCE_DIR}/src/backend # common headers
        )
endfunction()
