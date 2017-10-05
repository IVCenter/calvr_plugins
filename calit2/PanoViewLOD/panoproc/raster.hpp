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

#ifndef PANOPROC_RASTER_HPP
#define PANOPROC_RASTER_HPP

#include <string>
#include <unistd.h>
#include "sys/types.h"

//------------------------------------------------------------------------------

class raster
{
public:

    raster(int, int, int, int, bool, const std::string&, off_t);
    raster(int, int, int, int, bool);
   ~raster();
   
    int  getw() const { return w; }
    int  geth() const { return h; }
    int  getc() const { return c; }
    int  getb() const { return b; }
    bool gets() const { return s; }

    int   length(   ) const { return w * h * c * b / 8; }
    void *buffer(int);

    void  get(int, int, double *) const;
    void  put(int, int, const double *);
    
    void  hdiff();
    
    void accum_histogram(unsigned int *, int);
    void match_histogram(unsigned int *, int);
    
private:

    int    clampi(int)    const;
    int    clampj(int)    const;
    double clampf(double) const;

    template <typename T> void get(int, int, int, double *) const;
    template <typename T> void put(int, int, int, const double *);

    void   *p;
    int     w;
    int     h;
    int     c;
    int     b;    
    bool    s;

    int    fd;
    void  *fp;
    size_t fs;
};

//------------------------------------------------------------------------------

#endif
