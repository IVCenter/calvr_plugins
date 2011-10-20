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

#include "sampler_testpattern.hpp"

//------------------------------------------------------------------------------

static const double color[6][3] = {
    { 1.0, 0.0, 0.0 },
    { 1.0, 1.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 0.0, 1.0, 1.0 },
    { 0.0, 0.0, 1.0 },
    { 1.0, 0.0, 1.0 },
};

void sampler_testpattern::get(double *p, int i, int j,
                        const double *a, const double *b,
                        const double *c, const double *d)
{
    int l = 0;

    p[0] = color[l % 6][0];
    p[1] = color[l % 6][1];
    p[2] = color[l % 6][2];

    if (i % 32 < 16)
    {
        p[0] *= 0.5;
        p[1] *= 0.5;
        p[2] *= 0.5;
    }

    if (j % 32 < 16)
    {
        p[0] *= 0.5;
        p[1] *= 0.5;
        p[2] *= 0.5;
    }
}

//------------------------------------------------------------------------------
