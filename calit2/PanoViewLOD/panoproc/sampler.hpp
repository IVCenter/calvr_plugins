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

#ifndef PANOPROC_SAMPLER_HPP
#define PANOPROC_SAMPLER_HPP

#include "image.hpp"

//------------------------------------------------------------------------------

static inline void mid2(double *v, const double *a, const double *b)
{
    v[0] = (a[0] + b[0]) / 2;
    v[1] = (a[1] + b[1]) / 2;
    v[2] = (a[2] + b[2]) / 2;
}

static inline void mid4(double *v, const double *a, const double *b,
                                   const double *c, const double *d)
{
    v[0] = (a[0] + b[0] + c[0] + d[0]) / 4;
    v[1] = (a[1] + b[1] + c[1] + d[1]) / 4;
    v[2] = (a[2] + b[2] + c[2] + d[2]) / 4;
}

//------------------------------------------------------------------------------

static inline double min2(double a, double b)
{
    return (a < b) ? a : b;
}

static inline double min4(double a, double b, double c, double d)
{
    return min2(min2(a, b), min2(c, d));
}

//------------------------------------------------------------------------------

static inline double max2(double a, double b)
{
    return (a > b) ? a : b;
}

static inline double max4(double a, double b, double c, double d)
{
    return max2(max2(a, b), max2(c, d));
}

//------------------------------------------------------------------------------

class sampler
{
public:

    sampler() : source(0) { }

    void setimage(image *s) { source = s; }

    int  getc() const { return source->getc(); }
    int  getb() const { return source->getb(); }
    bool gets() const { return source->gets(); }
    
    virtual void get(double *p, const double *a, const double *b,
                                const double *c, const double *d)
    {
        p[0] = 0;
        p[1] = 0;
        p[2] = 0;
    }

    virtual void get(double *p, int i, int j, const double *a, const double *b,
                                              const double *c, const double *d)
    {
        return get(p, a, b, c, d);
    }

protected:

    image* source;
};

//------------------------------------------------------------------------------

#endif
