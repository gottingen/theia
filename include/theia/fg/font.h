// Copyright 2024 The Turbo Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once
#include <theia/fg/defines.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup font_functions
 *  @{
 */

/**
   Create a Font object

   \param[out] pFont will point to the font object created after this function returns

   \return \ref fg_err error code
 */
FGAPI fg_err fg_create_font(fg_font* pFont);

/**
   Increase reference count of the resource

   \param[out] pOut is the new handle to existing resource
   \param[in] pIn is the existing resource handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_retain_font(fg_font *pOut, fg_font pIn);

/**
   Destroy font object

   \param[in] pFont is the font handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_release_font(fg_font pFont);

/**
   Load a given font file

   \param[in] pFont is the font handle
   \param[in] pFileFullPath True Type Font file path

   \return \ref fg_err error code
 */
FGAPI fg_err fg_load_font_file(fg_font pFont, const char* const pFileFullPath);

/**
   Load a system font based on the name

   \param[in] pFont is the font handle
   \param[in] pFontName True Type Font name

   \return \ref fg_err error code
 */
FGAPI fg_err fg_load_system_font(fg_font pFont, const char* const pFontName);

/** @} */

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

namespace theia
{

/// \brief Font object is a resource handler for the font you want to use
class Font {
    private:
        fg_font mValue;

    public:
        /**
           Creates Font object
         */
        FGAPI Font();

        /**
           Copy constructor for Font

           \param[in] other is the Font object of which we make a copy of, this is not a deep copy.
         */
        FGAPI Font(const Font& other);

        /**
           Font Destructor
         */
        FGAPI ~Font();

        /**
           Load a given font file

           \param[in] pFile True Type Font file path
         */
        FGAPI void loadFontFile(const char* const pFile);

        /**
           Load a system font based on the name

           \param[in] pName True Type Font name
         */
        FGAPI void loadSystemFont(const char* const pName);

        /**
           Get handle for internal implementation of Font object
         */
        FGAPI fg_font get() const;
};

}

#endif
