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
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

const unsigned IMGW  = 256;
const unsigned IMGH  = 256;
const unsigned DIMX  = 1000;
const unsigned DIMY  = 800;
const unsigned NBINS = 256;

using namespace std;

struct Bitmap {
    unsigned char* ptr;
    unsigned width;
    unsigned height;
};

class PerlinNoise {
   private:
    float base[IMGW][IMGH];
    float perlin[IMGW][IMGH];

   public:
    PerlinNoise();
    float noise(float u, float v);
};

Bitmap createBitmap(unsigned w, unsigned h);

void destroyBitmap(Bitmap& bmp);

void kernel(Bitmap& bmp);

void populateBins(Bitmap& bmp, int* hist_array, const unsigned nbins,
                  float* hist_cols);

int main(int argc, char* argv[]) {
    Bitmap bmp = createBitmap(IMGW, IMGH);
    /*
     * First theia call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other theia::* object to be created successfully
     */
    theia::Window wnd(DIMX, DIMY, "Histogram Demo");
    wnd.makeCurrent();

    theia::Image img(IMGW, IMGH, FG_RGBA, theia::u8);

    theia::Chart chart(FG_CHART_2D);

    /* set x axis limits to maximum and minimum values of data
     * and y axis limits to range [0, number of pixels ideally]
     * but practically total number of pixels as y range will skew
     * the histogram graph vertically. Therefore setting it to
     * 25% of total number of pixels */
    chart.setAxesLimits(0, 1, 0, IMGW * IMGH / (float)(NBINS / 4.0));

    /*
     * Create histogram object specifying number of bins
     */
    theia::Histogram hist = chart.histogram(NBINS, theia::s32);
    /*
     * Set histogram colors
     */
    hist.setColor(FG_YELLOW);

    GfxHandle* handles[3];

    createGLBuffer(&handles[0], img.pixels(), THEIA_IMAGE_BUFFER);
    createGLBuffer(&handles[1], hist.vertices(), THEIA_VERTEX_BUFFER);
    createGLBuffer(&handles[2], hist.colors(), THEIA_VERTEX_BUFFER);

    wnd.setColorMap((fg_color_map)(argc == 2 ? atoi(argv[1]) : 1));

    do {
        /*
         * generate image, and prepare data to pass into
         * Histogram's underlying vertex buffer object
         */
        kernel(bmp);

        copyToGLBuffer(handles[0], (ComputeResourceHandle)bmp.ptr, img.size());

        // theia::copy(img, (const void*)bmp.ptr);

        /* copy your data into the vertex buffer object exposed by
         * theia::Histogram class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, theia provides copy headers
         * along with the library to help with this task
         */
        std::vector<int> histArray(NBINS, 0);
        std::vector<float> colArray(3 * NBINS, 0.0f);
        populateBins(bmp, histArray.data(), NBINS, colArray.data());

        copyToGLBuffer(handles[1], (ComputeResourceHandle)histArray.data(),
                       hist.verticesSize());
        copyToGLBuffer(handles[2], (ComputeResourceHandle)colArray.data(),
                       hist.colorsSize());

        /*
         * Split the window into grid regions
         */
        // wnd.draw(2, 2, 0, img,  "Dynamic Perlin Noise" );
        // wnd.draw(2, 2, 1, img,  "Dynamic Perlin Noise" );
        // wnd.draw(2, 1, 1, chart, "Histogram of Noisy Image");
        wnd.draw(2, 3, 0, img, "Dynamic Perlin Noise");
        wnd.draw(2, 3, 1, img, "Dynamic Perlin Noise");
        wnd.draw(2, 3, 2, img, "Dynamic Perlin Noise");
        wnd.draw(2, 2, 2, chart, "Histogram of Noisy Image");
        wnd.draw(2, 2, 3, chart, "Histogram of Noisy Image");

        wnd.swapBuffers();
    } while (!wnd.close());

    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);

    return 0;
}

float interp(float x0, float x1, float alpha) {
    return x0 * (1 - alpha) + alpha * x1;
}

PerlinNoise::PerlinNoise() {
    std::srand((unsigned)(std::time(0)));

    for (unsigned i = 0; i < IMGW; i++) {
        for (unsigned j = 0; j < IMGH; j++) {
            base[i][j]   = std::rand() / (float)(RAND_MAX);
            perlin[i][j] = 0;
        }
    }

    float persistence = 0.5f;
    float amp         = 1.0f;
    float tamp        = 0.0f;

    for (int octave = 6; octave >= 0; --octave) {
        int period = 1 << octave;
        float freq = 1.0f / period;

        for (unsigned i = 0; i < IMGW; i++) {
            int si0      = (i / period) * period;
            int si1      = (si0 + period) % IMGW;
            float hblend = (i - si0) * freq;

            for (unsigned j = 0; j < IMGH; j++) {
                int sj0      = (j / period) * period;
                int sj1      = (sj0 + period) % IMGH;
                float vblend = (j - sj0) * freq;

                float top = interp(base[si0][sj0], base[si1][sj0], hblend);
                float bot = interp(base[si0][sj1], base[si1][sj1], hblend);

                perlin[i][j] += (amp * interp(top, bot, vblend));
            }
        }
        tamp += amp;
        amp *= persistence;
    }

    for (unsigned i = 0; i < IMGW; i++)
        for (unsigned j = 0; j < IMGH; j++) perlin[i][j] /= tamp;
}

float PerlinNoise::noise(float u, float v) {
    return perlin[(unsigned)(IMGW * u)][(unsigned)(IMGH * v)];
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
    PerlinNoise perlin;

    for (unsigned y = 0; y < bmp.height; ++y) {
        for (unsigned x = 0; x < bmp.width; ++x) {
            int offset = x + y * bmp.width;

            float u = x / (float)(bmp.width);
            float v = y / (float)(bmp.height);

            unsigned char noiseVal  = (unsigned char)(255 * perlin.noise(u, v));
            bmp.ptr[offset * 4 + 0] = noiseVal;
            bmp.ptr[offset * 4 + 1] = noiseVal;
            bmp.ptr[offset * 4 + 2] = noiseVal;
            bmp.ptr[offset * 4 + 3] = 255;
        }
    }
}

void populateBins(Bitmap& bmp, int* hist_array, const unsigned nbins,
                  float* hist_cols) {
    for (unsigned y = 0; y < bmp.height; ++y) {
        for (unsigned x = 0; x < bmp.width; ++x) {
            int offset             = x + y * bmp.width;
            unsigned char noiseVal = bmp.ptr[offset * 4];
            unsigned idx           = (int)((float)noiseVal / 255.f * nbins);
            hist_array[idx]++;
        }
    }

    for (unsigned b = 0; b < nbins; ++b) {
        hist_cols[3 * b + 0] = std::rand() / (float)RAND_MAX;
        hist_cols[3 * b + 1] = std::rand() / (float)RAND_MAX;
        hist_cols[3 * b + 2] = std::rand() / (float)RAND_MAX;
    }
}
