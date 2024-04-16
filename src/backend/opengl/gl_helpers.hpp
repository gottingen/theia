/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <theia/fg/defines.h>
#include <theia/glad/glad.h>

namespace theia {
namespace opengl {

/* Convert theia type enum to OpenGL enum for GL_* type
 *
 * @pValue is the theia type enum
 *
 * @return GL_* typedef for data type
 */
GLenum dtype2gl(const theia::dtype pValue);

/* Convert theia channel format enum to OpenGL enum to indicate color component
 * layout
 *
 * @pValue is the theia type enum
 *
 * @return OpenGL enum indicating color component layout
 */
GLenum ctype2gl(const theia::ChannelFormat pMode);

/* Convert theia channel format enum to OpenGL enum to indicate color component
 * layout
 *
 * This function is used to group color component layout formats based
 * on number of components.
 *
 * @pValue is the theia type enum
 *
 * @return OpenGL enum indicating color component layout
 */
GLenum ictype2gl(const theia::ChannelFormat pMode);

/* Create OpenGL buffer object
 *
 * @pTarget should be either GL_ARRAY_BUFFER or GL_ELEMENT_ARRAY_BUFFER
 * @pSize is the size of the data in bytes
 * @pPtr is the pointer to host data. This can be NULL
 * @pUsage should be either GL_STATIC_DRAW or GL_DYNAMIC_DRAW
 *
 * @return OpenGL buffer object identifier
 */
template<typename T>
GLuint createBuffer(GLenum pTarget, size_t pSize, const T* pPtr,
                    GLenum pUsage) {
    GLuint retVal = 0;
    glGenBuffers(1, &retVal);
    glBindBuffer(pTarget, retVal);
    glBufferData(pTarget, pSize * sizeof(T), pPtr, pUsage);
    glBindBuffer(pTarget, 0);
    return retVal;
}

/* Get a vertex buffer object for quad that spans the screen
 */
GLuint screenQuadVBO(const int pWindowId);

/* Get a vertex array object that uses screenQuadVBO
 *
 * This vertex array object when bound and rendered, basically
 * draws a rectangle over the entire screen with standard
 * texture coordinates. Use of this vao would be as follows
 *
 *     `glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);`
 */
GLuint screenQuadVAO(const int pWindowId);

void glErrorCheck(const char* pMsg, const char* pFile, int pLine);

}  // namespace opengl
}  // namespace theia

#define CheckGL(msg) glErrorCheck(msg, __FILE__, __LINE__)
