// Copyright (c) 2011 Robert Kooima
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <cstdio>
#include <sys/time.h>

#include "image_basic.hpp"
#include "error.hpp"

//------------------------------------------------------------------------------

image_basic::image_basic(const char *arg) : w(0), h(0), c(0), b(0), R(1), C(1)
{
    int rr;
    int cc;
    
    if (sscanf(arg, "%dx%d", &rr, &cc) == 2)
    {
        R = rr;
        C = cc;
    }
}

image_basic::~image_basic()
{
    std::vector<raster *>::iterator i;
    
    for (i = data.begin(); i != data.end(); ++i)
        delete (*i);
}

//------------------------------------------------------------------------------

void image_basic::source(const std::string& name)
{
    raster *d = 0;
        
    // Load the named file. Report the time taken to do so.
    
    struct timeval t0;
    struct timeval t1;
    
    gettimeofday(&t0, 0);
    {
        std::string ext(name, name.rfind('.') + 1);

        if      (ext == "tif" || ext == "TIF") d = load_tif(name);
        else if (ext == "png" || ext == "PNG") d = load_png(name);
        else if (ext == "jpg" || ext == "JPG") d = load_jpg(name);
    }
    gettimeofday(&t1, 0);
        
    printf("%6.2lfs : %s\n", (t1.tv_sec  - t0.tv_sec) +
                 0.0000001 * (t1.tv_usec - t0.tv_usec), name.c_str());

    // Store the image and confirm that all images have the same format.
    
    if (d)
    {
        data.push_back(d);

        if (data.size() == 1)
        {
            w = data.back()->getw();
            h = data.back()->geth();
            c = data.back()->getc();
            b = data.back()->getb();
        }
        else
        {
            if (data.back()->getw() != w ||
                data.back()->geth() != h ||
                data.back()->getc() != c ||
                data.back()->getb() != b)
                app_err_exit("%s format mismatch.", name.c_str());
        }
    }
}

//------------------------------------------------------------------------------

static inline int max(int a, int b)
{
    return (a > b) ? a : b;
}

static inline int min(int a, int b)
{
    return (a < b) ? a : b;
}

void image_basic::tap(int H, int i, int j, double *p) const
{
    i = min(max(i, 0), h * R - 1);
    j = min(max(j, 0), w * C - 1);
    
    const int rr = i / h, ii = i % h;
    const int cc = j / w, jj = j % w;

    data[rr * C + cc]->get(ii, jj, p);
}

//------------------------------------------------------------------------------
