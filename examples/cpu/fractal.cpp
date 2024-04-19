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

#include <theia/theia.h>
#define USE_THEIA_CPU_COPY_HELPERS
#include <theia/fg/compute_copy.h>
#include <cmath>
#include <complex>

const unsigned DIMX = 512;
const unsigned DIMY = 512;

struct Bitmap {
    unsigned char* ptr;
    unsigned width;
    unsigned height;
};

Bitmap createBitmap(unsigned w, unsigned h);
void destroyBitmap(Bitmap& bmp);
void kernel(Bitmap& bmp);
int julia(int x, int y, int width, int height);

int main(void) {
    Bitmap bmp = createBitmap(DIMX, DIMY);

    /*
     * First theia call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other theia::* object to be created successfully
     */
    theia::Window wnd(DIMX, DIMY, "Fractal Demo");
    wnd.makeCurrent();

    /* create an font object and load necessary font
     * and later pass it on to window object so that
     * it can be used for rendering text
     *
     * NOTE: THIS IS OPTIONAL STEP, BY DEFAULT WINDOW WILL
     * HAVE FONT ALREADY SETUP*/
    theia::Font fnt;
#if defined(OS_WIN)
    fnt.loadSystemFont("Calibri");
#else
    fnt.loadSystemFont("Vera");
#endif
    wnd.setFont(&fnt);

    /* Create an image object which creates the necessary
     * textures and pixel buffer objects to hold the image
     * */
    theia::Image img(DIMX, DIMY, FG_RGBA, theia::u8);
    /* copy your data into the pixel buffer object exposed by
     * theia::Image class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, theia provides copy headers
     * along with the library to help with this task
     */
    kernel(bmp);

    GfxHandle* handle = 0;

    // create GL-CPU interop buffer
    createGLBuffer(&handle, img.pixels(), THEIA_IMAGE_BUFFER);

    // copy the data from compute buffer to graphics buffer
    copyToGLBuffer(handle, (ComputeResourceHandle)bmp.ptr, img.size());

    do { wnd.draw(img); } while (!wnd.close());

    // destroy GL-CPU Interop buffer
    releaseGLBuffer(handle);
    destroyBitmap(bmp);
    return 0;
}

Bitmap createBitmap(unsigned w, unsigned h) {
    Bitmap retVal;
    retVal.width  = w;
    retVal.height = h;
    retVal.ptr    = new unsigned char[4 * w * h];
    return retVal;
}

void destroyBitmap(Bitmap& bmp) { delete[] bmp.ptr; }

void kernel(Bitmap& bmp) {
    for (unsigned y = 0; y < bmp.height; ++y) {
        for (unsigned x = 0; x < bmp.width; ++x) {
            int offset              = x + y * bmp.width;
            int juliaVal            = julia(x, y, bmp.width, bmp.height);
            bmp.ptr[offset * 4 + 0] = 255 * juliaVal;
            bmp.ptr[offset * 4 + 1] = 0;
            bmp.ptr[offset * 4 + 2] = 0;
            bmp.ptr[offset * 4 + 3] = 255;
        }
    }
}

int julia(int x, int y, int width, int height) {
    const float scale = 1.5;
    float jx          = scale * (float)(width / 2.0f - x) / (width / 2.0f);
    float jy          = scale * (float)(height / 2.0f - y) / (height / 2.0f);

    std::complex<float> c(-0.8f, 0.156f);
    std::complex<float> a(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (abs(a) > 1000) return 0;
    }

    return 1;
}
