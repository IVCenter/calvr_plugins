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

#include "sampler_adaptive.hpp"

//------------------------------------------------------------------------------

static inline double contrast(double a, double b, double c, double d)
{
    const double min = min4(a, b, c, d);
    const double max = max4(a, b, c, d);

    return (max - min) / (max + min);
}

void sampler_adaptive::adapt(double *p, const double *a, const double *b,
                                        const double *c, const double *d, int r)
{
    double v[3];

    double va[3], pa[4];
    double vb[3], pb[4];
    double vc[3], pc[4];
    double vd[3], pd[4];

    mid4(v, a, b, c, d);
    mid2(va, v, a);
    mid2(vb, v, b);
    mid2(vc, v, c);
    mid2(vd, v, d);

    source->get(pa, va);
    source->get(pb, vb);
    source->get(pc, vc);
    source->get(pd, vd);

    if (r < 4)
    {
        if (contrast(pa[0], pb[0], pc[0], pd[0]) > kr ||
            contrast(pa[1], pb[1], pc[1], pd[1]) > kg ||
            contrast(pa[2], pb[2], pc[2], pd[2]) > kb)
        {
            double n[3];
            double s[3];
            double e[3];
            double w[3];

            mid2(n, a, b);
            mid2(s, c, d);
            mid2(e, a, c);
            mid2(w, b, d);

            adapt(pa, a, n, e, v, r + 1);
            adapt(pb, n, b, v, w, r + 1);
            adapt(pc, e, v, c, s, r + 1);
            adapt(pd, v, w, s, d, r + 1);
        }
    }

    switch (source->getc())
    {
        case 4: p[3] = (pa[3] + pb[3] + pc[3] + pd[3]) / 4;
        case 3: p[2] = (pa[2] + pb[2] + pc[2] + pd[2]) / 4;
        case 2: p[1] = (pa[1] + pb[1] + pc[1] + pd[1]) / 4;
        case 1: p[0] = (pa[0] + pb[0] + pc[0] + pd[0]) / 4;
    }
}

void sampler_adaptive::get(double *p, const double *a, const double *b,
                                      const double *c, const double *d)
{
    adapt(p, a, b, c, d, 0);
}

//------------------------------------------------------------------------------
