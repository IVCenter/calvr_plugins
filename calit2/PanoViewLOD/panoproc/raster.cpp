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

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <stdint.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "raster.hpp"
#include "error.hpp"

//------------------------------------------------------------------------------

raster::raster(int w, int h, int c, int b, bool s,
               const std::string& name, off_t o)
    : w(w), h(h), c(c), b(b), s(s), fd(0), fp(0), fs(0)
{
    struct stat st;
    int         fl = MAP_PRIVATE; // |= MAP_POPULATE;

    if ((stat(name.c_str(), &st)) != -1 && (fs = st.st_size))
    {
        if ((fd = open(name.c_str(), O_RDONLY)) != -1)
        {
            if ((fp = mmap(0, fs, PROT_READ, fl, fd, 0)) != MAP_FAILED)
            {
                p = (char *) fp + o;
            }
            else sys_err_mesg("mmap %s", name.c_str());
        }
        else sys_err_mesg("open %s", name.c_str());
    }
    else sys_err_mesg("stat %s", name.c_str());
}


raster::raster(int w, int h, int c, int b, bool s)
    : w(w), h(h), c(c), b(b), s(s), fd(0), fp(0), fs(0)
{
    p = calloc(size_t(w) * size_t(h) * size_t(c), b / 8);
}

raster::~raster()
{
    if (fp)
    {
        munmap(fp, fs);
        close (fd);
    }
    else
        free(p);
}

void *raster::buffer(int i)
{
    return (char *) p + size_t(w) * size_t(c) * size_t(i) * size_t(b / 8);
}

//------------------------------------------------------------------------------

int raster::clampi(int i) const
{
    if      (i <     0) return     0;
    else if (i > h - 1) return h - 1;
    else                return i;
}

int raster::clampj(int j) const
{
    if      (j <     0) return     0;
    else if (j > w - 1) return w - 1;
    else                return j;
}

double raster::clampf(double f) const
{
    if      ( s && f < -1) return -1;
    else if (!s && f <  0) return  0;
    else if (      f >  1) return  1;
    else                   return  f;
}

template <typename T> void raster::get(int i, int j, int m, double *dst) const
{
    const T *src = (const T *) p + size_t(c) * (size_t(w) * i + j);

    switch (c)
    {
        case 4: dst[3] = double(src[3]) / double(m);
        case 3: dst[2] = double(src[2]) / double(m);
        case 2: dst[1] = double(src[1]) / double(m);
        case 1: dst[0] = double(src[0]) / double(m);
    }
}

template <typename T> void raster::put(int i, int j, int m, const double *src)
{
    T *dst = (T *) p + size_t(c) * (size_t(w) * i + j);

    switch (c)
    {
        case 4: dst[3] = T(clampf(src[3]) * double(m));
        case 3: dst[2] = T(clampf(src[2]) * double(m));
        case 2: dst[1] = T(clampf(src[1]) * double(m));
        case 1: dst[0] = T(clampf(src[0]) * double(m));
    }
}

void raster::get(int i, int j, double *dst) const
{
    assert(p);
    
    i = clampi(i);
    j = clampj(j);
    
    switch (b)
    {
        case 8:
        {
            if (s) get<char>          (i, j,   127, dst);
            else   get<unsigned char> (i, j,   255, dst);
            break;
        }
        case 16:
        {
            if (s) get<short>         (i, j, 32767, dst);
            else   get<unsigned short>(i, j, 65535, dst);
            break;
        }
        case 32:
        {
            get<float>(i, j, 1.f, dst);
            break;
        }
    }
}

void raster::put(int i, int j, const double *src)
{
    assert(p);
    assert(fs == 0);
    assert(fd == 0);
    
    i = clampi(i);
    j = clampj(j);
    
    switch (b)
    {
        case 8:
        {
            if (s) put<char>          (i, j,   127, src);
            else   put<unsigned char> (i, j,   255, src);
            break;
        }
        case 16:
        {
            if (s) put<short>         (i, j, 32767, src);
            else   put<unsigned short>(i, j, 65535, src);
            break;
        }
        case 32:
        {
            put<float>(i, j, 1.f, src);
            break;
        }
    }
}

//------------------------------------------------------------------------------

// Accumulate the histogram of channel k. Assume ptr points to a buffer of 2^b
// values, one for each possible pixel value.  Ignore b=32 for now.

void raster::accum_histogram(unsigned int *H, int k)
{
    switch (b)
    {
        case 8:
        {
            const uint8_t *src = (const uint8_t *) p;
            
            for (size_t i = 0; i < size_t(w) * size_t(h); ++i)
                H[src[i * c + k]]++;
            
            break;
        }
        case 16:
        {
            const uint16_t *src = (const uint16_t *) p;
            
            for (size_t i = 0; i < size_t(w) * size_t(h); ++i)
                H[src[i * c + k]]++;
            
            break;
        }
    }
}

#if 0
void raster::match_histogram(unsigned int *H, int k)
{
    if (b == 16)
    {
        int m = 1 << b;
        int n = w * h;
    
        // Allocate storage for this histogram and both cumulative histograms.
    
        unsigned int *G  = (unsigned int *) calloc(m, sizeof (unsigned int));
        unsigned int *GG = (unsigned int *) calloc(m, sizeof (unsigned int));
        unsigned int *HH = (unsigned int *) calloc(m, sizeof (unsigned int));
        unsigned int *M  = (unsigned int *) calloc(m, sizeof (unsigned int));

        if (G && GG && HH && M)
        {
            // Calculate the histogram of this image.
        
            accum_histogram(G, k);
            
            // Calculate cumulative values of both histograms.
            
            GG[0] = G[0];
            HH[0] = H[0];
            
            for (int i = 1; i < m; ++i)
            {
                GG[i] = GG[i - 1] + G[i];
                HH[i] = HH[i - 1] + H[i];
            }
            
            // Determine the mapping from GG to HH.
            
            for (int i = 0, j = 0; i < m && j < m; )
            {
                double g = double(GG[i]) / double(GG[m - 1]);
                double h = double(HH[j]) / double(HH[m - 1]);
                
                if (g > h) M[i] = j++;
                else       M[i++] = j;
            }

            // Apply the mapping to this image.

            uint16_t *src = (uint16_t *) p;
            uint16_t *dst = (uint16_t *) calloc(n, c * b);
            
            for (int i = 0; i < n; ++i)
                dst[i * c + k] = uint16_t(M[src[i * c + k]]);

            // HACK: This LEAKS a HUGE amount of memory.

            p = dst;
        }
        
        free(M);
        free(HH);
        free(GG);
        free(G);
    }
}
#endif
//------------------------------------------------------------------------------

// Apply a TIFF-standard horizontal difference prediction to this raster.

void raster::hdiff()
{
    assert(p);
    
    const int n = w * c;
    
    switch (b)
    {
        case 8:
        {
            char  *dst = (char  *) p;
            
            for     (int i =     0; i <  h; ++i)
                for (int j = n - 1; j >= c; --j)
                    dst[n * i + j] -= dst[n * i + j - c];
            break;
        }
        case 16:
        {
            short *dst = (short *) p;
            
            for     (int i =     0; i <  h; ++i)
                for (int j = n - 1; j >= c; --j)
                    dst[n * i + j] -= dst[n * i + j - c];
            break;
        }
        case 32:
        {
            float *dst = (float *) p;
            
            for     (int i =     0; i <  h; ++i)
                for (int j = n - 1; j >= c; --j)
                    dst[n * i + j] -= dst[n * i + j - c];
            break;
        }
    }
}

//------------------------------------------------------------------------------
