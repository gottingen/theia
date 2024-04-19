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

function(conditional_directory variable directory)
    if(${variable})
        add_subdirectory(${directory})
    endif()
endfunction()
