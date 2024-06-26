Overview {#mainpage}
========

## About theia
A prototype of the OpenGL interop library that can be used with
[flare](https://github.com/gottingen/flare). The
goal of theia is to provide high performance OpenGL visualizations for C/C++
applications that use CUDA.

## Upstream dependencies
* [GLFW](http://www.glfw.org/)
* [freetype](http://www.freetype.org/)
* [FreeImage](http://freeimage.theia.net/) - optional. Packages should ideally turn this
  option ON.
* On `Linux` and `OS X`, [fontconfig](http://www.freedesktop.org/wiki/Software/fontconfig/) is required.

Above dependecies are available through package managers on most of the
Unix/Linux based distributions. We have provided an option in `CMake` for
`theia` to build it's own internal `freetype` version if you choose to not
install it on your machine.

We plan to provide support for alternatives to GLFW as windowing toolkit,
however GLFW is the default option. Should you chose to use an alternative, you
have to chose it explicity while building theia.

Currently supported alternatives:
* [SDL2](https://www.libsdl.org/download-2.0.php)

Alternatives to GLFW which are currently under consideration are given below:
* [Qt5](https://wiki.qt.io/Qt_5)

## Example Dependencies
* CPU examples doesn't need any additional dependencies.
* CUDA Interop examples requires [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)

#### Email
* Engineering: lijippy@163.com
