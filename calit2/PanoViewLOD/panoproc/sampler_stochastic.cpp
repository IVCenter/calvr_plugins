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

#include "sampler_stochastic.hpp"

//------------------------------------------------------------------------------

static inline double lerp(double a, double b, double t)
{
    return a * (1 - t) + b * t;
}

void sampler_stochastic::get(double *p, int i, int j, int k,
                       const double *a, const double *b,
                       const double *c, const double *d)
{
    double v[3];
    float  s[2];

    point.get_p(i, j, k, s);

    v[0] = lerp(lerp(a[0], b[0], s[0]), lerp(c[0], d[0], s[0]), s[1]);
    v[1] = lerp(lerp(a[1], b[1], s[0]), lerp(c[1], d[1], s[0]), s[1]);
    v[2] = lerp(lerp(a[2], b[2], s[0]), lerp(c[2], d[2], s[0]), s[1]);

    source->get(p, v);
}

void sampler_stochastic::get(double *p, int i, int j,
                       const double *a, const double *b,
                       const double *c, const double *d)
{
    int k = 0, n = point.get_n(i, j);

    double min[3] = { 1, 1, 1 };
    double max[3] = { 0, 0, 0 };

    double t[3];

    p[0] = 0;
    p[1] = 0;
    p[2] = 0;

    // Accumulate the first few samples and note the extremes.

    while (k < 8)
    {
        get(t, i, j, k, a, b, c, d);

        min[0] = min2(min[0], t[0]);
        min[1] = min2(min[1], t[1]);
        min[2] = min2(min[2], t[2]);

        max[0] = max2(max[0], t[0]);
        max[1] = max2(max[1], t[1]);
        max[2] = max2(max[2], t[2]);

        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];

        k++;
    }

    // If the contrast exceeds the limit, accumulate the remaining samples.

    if (0.4 < (max[0] - min[0]) / (max[0] + min[0]) ||
        0.3 < (max[1] - min[1]) / (max[1] + min[1]) ||
        0.6 < (max[2] - min[2]) / (max[2] + min[2]))

        while (k < n)
        {
            get(t, i, j, k, a, b, c, d);

            p[0] += t[0];
            p[1] += t[1];
            p[2] += t[2];

            k++;
        }

    // Average the accumulated samples.

    p[0] /= k;
    p[1] /= k;
    p[2] /= k;
}

//------------------------------------------------------------------------------
