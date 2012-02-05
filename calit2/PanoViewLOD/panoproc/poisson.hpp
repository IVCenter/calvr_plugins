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

#ifndef PANOPROC_POISSON_HPP
#define PANOPROC_POISSON_HPP

#include <vector>

//------------------------------------------------------------------------------

class poisson
{
public:

    poisson(int m, float r, float k);

    void get_p(int i, int j, int k, float *p);
    int  get_n(int i, int j);

private:

    // Poisson distribution generators.

    bool test1(int i, int j, float x, float y, float d);
    bool test9(int i, int j, float x, float y, float d);

    void fill(int m, float r, float k);
    void dump();

    // 2D point data type.

    struct point
    {
        float x;
        float y;
    };

    typedef std::vector<point>           point_v;
    typedef std::vector<point>::iterator point_i;

    // Grid of bins of points.

    static const int grid_w = 32;
    static const int grid_h = 32;
    static const int grid_n = 32;

    point_v grid[grid_w][grid_h];
};

//------------------------------------------------------------------------------

#endif
