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
#include <cstdio>
#include <cmath>

#include "poisson.hpp"

//------------------------------------------------------------------------------

poisson::poisson(int m, float r, float k)
{
    srand48(0);
    fill(m, r, k);
}

//------------------------------------------------------------------------------

static inline int mod(int a, int n)
{
    return (a % n < 0) ? (a % n + n) : (a % n);
}

void poisson::get_p(int i, int j, int k, float *p)
{
    int ii = mod(i, grid_h);
    int jj = mod(j, grid_w);

    p[0] = grid[ii][jj][k].x * grid_w - jj;
    p[1] = grid[ii][jj][k].y * grid_h - ii;
}

int poisson::get_n(int i, int j)
{
    int ii = mod(i, grid_h);
    int jj = mod(j, grid_w);

    return int(grid[ii][jj].size());
}

//------------------------------------------------------------------------------

// Generate a poisson distribution of n points on a torroidal domain using the
// "Hierarchical Poisson Disk Sampling Distributions" approach of McCool and
// Fiume, Graphics Interface 1992. Begin with a radius of r and allow up to m
// point placement failures before relaxing the radius by a factor of k.

bool poisson::test1(int i, int j, float x, float y, float d)
{
    if      (i ==     -1) { i += grid_h; y += 1.f; }
    else if (i == grid_h) { i -= grid_h; y -= 1.f; }
    if      (j ==     -1) { j += grid_w; x += 1.f; }
    else if (j == grid_w) { j -= grid_w; x -= 1.f; }

    for (point_i p = grid[i][j].begin(); p != grid[i][j].end(); ++p)
    {
        float dx = x - p->x;
        float dy = y - p->y;

        if (dx * dx + dy * dy < d * d)
            return false;
    }
    return true;
}

bool poisson::test9(int i, int j, float x, float y, float d)
{
    return (test1(i - 1, j - 1, x, y, d) &&
            test1(i - 1, j,     x, y, d) &&
            test1(i - 1, j + 1, x, y, d) &&
            test1(i,     j - 1, x, y, d) &&
            test1(i,     j,     x, y, d) &&
            test1(i,     j + 1, x, y, d) &&
            test1(i + 1, j - 1, x, y, d) &&
            test1(i + 1, j,     x, y, d) &&
            test1(i + 1, j + 1, x, y, d));
}

void poisson::fill(int m, float r, float k)
{
    int e = m;

    for (int c = 0; c < grid_w * grid_h * grid_n; ++c)
    {
        while (1)
        {
            point p;

            p.x = float(drand48());
            p.y = float(drand48());

            int i = int(floorf(p.y * grid_h));
            int j = int(floorf(p.x * grid_w));

            if (test9(i, j, p.x, p.y, r * 2))
            {
                grid[i][j].push_back(p);
                break;
            }

            if (--e == 0)
            {
                e  = m;
                r *= k;
            }
        }
    }
}

void poisson::dump()
{
    for     (int i = 0; i < grid_h; ++i)
        for (int j = 0; j < grid_w; ++j)
            for (point_i p = grid[i][j].begin(); p != grid[i][j].end(); ++p)

                printf("%f %f\n", p->x, p->y);
}

//------------------------------------------------------------------------------
