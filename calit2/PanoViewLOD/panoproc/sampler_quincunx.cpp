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

#include "sampler_quincunx.hpp"

//------------------------------------------------------------------------------

void sampler_quincunx::get(double *p, const double *a, const double *b,
                                      const double *c, const double *d)
{
    double M[3];
    double A[3];
    double B[3];
    double C[3];
    double D[3];

    mid4(M, a, b, c, d);
    mid2(A, a, M);
    mid2(B, b, M);
    mid2(C, c, M);
    mid2(D, d, M);

    double pm[4];
    double pa[4];
    double pb[4];
    double pc[4];
    double pd[4];

    source->get(pm, M);
    source->get(pa, A);
    source->get(pb, B);
    source->get(pc, C);
    source->get(pd, D);

    switch (source->getc())
    {
        case 4: p[3] = (pm[3] + pa[3] + pb[3] + pc[3] + pd[3]) / 5;
        case 3: p[2] = (pm[2] + pa[2] + pb[2] + pc[2] + pd[2]) / 5;
        case 2: p[1] = (pm[1] + pa[1] + pb[1] + pc[1] + pd[1]) / 5;
        case 1: p[0] = (pm[0] + pa[0] + pb[0] + pc[0] + pd[0]) / 5;
    }
}

//------------------------------------------------------------------------------
