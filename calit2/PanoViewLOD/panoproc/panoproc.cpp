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

#include <cstring>
#include <cmath>
#include <list>

#include <sys/time.h>

#include "panoproc.hpp"
#include "tiff.hpp"
#include "raster.hpp"

//------------------------------------------------------------------------------

#define NORM3 0.5773502691896258

const double page_v[8][3] = {
    {  NORM3,  NORM3,  NORM3 },
    { -NORM3,  NORM3,  NORM3 },
    {  NORM3, -NORM3,  NORM3 },
    { -NORM3, -NORM3,  NORM3 },
    {  NORM3,  NORM3, -NORM3 },
    { -NORM3,  NORM3, -NORM3 },
    {  NORM3, -NORM3, -NORM3 },
    { -NORM3, -NORM3, -NORM3 },
};

const int page_i[6][4] = {
    { 0, 4, 2, 6 },
    { 5, 1, 7, 3 },
    { 5, 4, 1, 0 },
    { 3, 2, 7, 6 },
    { 1, 0, 3, 2 },
    { 4, 5, 6, 7 },
};


struct turn
{
    unsigned int f : 3;
    unsigned int i : 3;
    unsigned int j : 3;
};

static const struct turn uu[6] = {
    { 2, 5, 3 }, { 2, 2, 0 }, { 5, 0, 5 },
    { 4, 3, 2 }, { 2, 3, 2 }, { 2, 0, 5 },
};

static const struct turn dd[6] = {
    { 3, 2, 3 }, { 3, 5, 0 }, { 4, 0, 2 },
    { 5, 3, 5 }, { 3, 0, 2 }, { 3, 3, 5 },
};

static const struct turn rr[6] = {
    { 4, 1, 3 }, { 5, 1, 3 }, { 1, 0, 1 },
    { 1, 3, 4 }, { 1, 1, 3 }, { 0, 1, 3 },
};

static const struct turn ll[6] = {
    { 5, 1, 0 }, { 4, 1, 0 }, { 0, 0, 4 },
    { 0, 3, 1 }, { 0, 1, 0 }, { 1, 1, 0 },
};

static int page_turn(const struct turn *tt, int f, int& i, int& j, int n)
{
    int o[6];

    o[0] = 0;
    o[1] = i;
    o[2] = j;
    o[3] = n     - 1;
    o[4] = n - i - 1;
    o[5] = n - j - 1;

    i  = o[tt[f].i];
    j  = o[tt[f].j];
    return tt[f].f;
}

//------------------------------------------------------------------------------

static void slerp1(double *a, const double *b, const double *c, double t)
{
    const double k = acos(b[0] * c[0] + b[1] * c[1] + b[2] * c[2]);
    const double u = sin(k - t * k) / sin(k);
    const double v = sin(    t * k) / sin(k);

    a[0] = b[0] * u + c[0] * v;
    a[1] = b[1] * u + c[1] * v;
    a[2] = b[2] * u + c[2] * v;
}

static void slerp2(double *v, const double *a, const double *b,
                              const double *c, const double *d,
                              double x, double y)
{
    double t[3];
    double u[3];

    slerp1(t, a, b, x);
    slerp1(u, c, d, x);
    slerp1(v, t, u, y);
}

//------------------------------------------------------------------------------

class page;

typedef std::list<page *> pagelist;

class page
{
public:

    page(off_t p, int i, int f, int d, int s, int x, int y) :
             p(p),  i(i),  f(f),  d(d),  s(s),  x(x),  y(y)  { }

    void sample(sampler *source, raster *data);
    void divide(pagelist& queue, int m, off_t p);

    const off_t p;                  // parent offset (position in the output)
    const int   i;                  // parent index  (which of four children)

protected:

    const int f;                    // mipmap root face
    const int d;                    // mipmap page level
    const int s;                    // mipmap page image size
    const int x;                    // mipmap X position of first pixel
    const int y;                    // mipmap Y position of first pixel
};

//------------------------------------------------------------------------------

/// Compute a pixel value as the average of three of its neighbors.

static void fx_pixel(raster *dst, int i, int j, int di, int dj)
{
    double A[4] = { 0, 0, 0, 0 };
    double B[4] = { 0, 0, 0, 0 };
    double C[4] = { 0, 0, 0, 0 };
    double D[4] = { 0, 0, 0, 0 };

    dst->get(i + di, j + dj, A);
    dst->get(i + di, j,      B);
    dst->get(i,      j + dj, C);

    switch (dst->getc())
    {
        case 4: D[3] = (A[3] + B[3] + C[3]) / 3.0;
        case 3: D[2] = (A[2] + B[2] + C[2]) / 3.0;
        case 2: D[1] = (A[1] + B[1] + C[1]) / 3.0;
        case 1: D[0] = (A[0] + B[0] + C[0]) / 3.0;
    }
    
    dst->put(i, j, D);
}


static void do_pixel(sampler *src,
                     raster  *dst,
                     int f, int d, int s, int x, int y, int r, int c)
{
    const int n = s * (1 << d);

    int g = f;
    int i = y + r;
    int j = x + c;

    // Wrap around a cube map edge if necessary.

    if      (j >= n) { g = page_turn(ll, f, i, j, n); }
    else if (j <  0) { g = page_turn(rr, f, i, j, n); }
    if      (i >= n) { g = page_turn(dd, f, i, j, n); }
    else if (i <  0) { g = page_turn(uu, f, i, j, n); }

    // Determine corner vectors for the current sample area.

    const double *A = page_v[page_i[g][0]];
    const double *B = page_v[page_i[g][1]];
    const double *C = page_v[page_i[g][2]];
    const double *D = page_v[page_i[g][3]];

    double ca[3];
    double cb[3];
    double cc[3];
    double cd[3];

    slerp2(ca, A, B, C, D, double(j)     / n, double(i)     / n);
    slerp2(cb, A, B, C, D, double(j + 1) / n, double(i)     / n);
    slerp2(cc, A, B, C, D, double(j)     / n, double(i + 1) / n);
    slerp2(cd, A, B, C, D, double(j + 1) / n, double(i + 1) / n);

    // Sample the area.

    double p[4];

    src->get(p, i, j, ca, cb, cc, cd);
    dst->put(r + 1, c + 1, p);
}

//------------------------------------------------------------------------------

/// Fill a data buffer for this page using the given sampler.

void page::sample(sampler *src, raster *dst)
{
    // Process each pixel of this page in turn.

    int r;
    int c;

    #pragma omp parallel for private(c)

    for     (r = -1; r <= s; ++r)
        for (c = -1; c <= s; ++c)
            do_pixel(src, dst, f, d, s, x, y, r, c);

    // Patch up the four corner pixels as needed.

    const int n = s * (1 << d);
    
    if (y     == 0 && x     == 0) fx_pixel(dst,     0,     0,  1,  1);
    if (y     == 0 && x + s == n) fx_pixel(dst,     0, s + 1,  1, -1);
    if (y + s == n && x + s == n) fx_pixel(dst, s + 1, s + 1, -1, -1);
    if (y + s == n && x     == 0) fx_pixel(dst, s + 1,     0, -1,  1);
}

/// Sub-divide this page.

void page::divide(pagelist& queue, int m, off_t p)
{
    if (d < m)
    {
        const int L = 1 + d;
        const int X = 2 * x;
        const int Y = 2 * y;

        queue.push_back(new page(p, 0, f, L, s, X,     Y));
        queue.push_back(new page(p, 1, f, L, s, X + s, Y));
        queue.push_back(new page(p, 2, f, L, s, X,     Y + s));
        queue.push_back(new page(p, 3, f, L, s, X + s, Y + s));
    }
}

//------------------------------------------------------------------------------

// Report the current progress, having finished i of n pages in t seconds.
// Extrapolate the time remaining until completion.

static void report(int n, int i, double t)
{
    static char str[128] = "";

    memset(str, '\b',  strlen(str));

    printf("%s", str);
    {
        const double s = i ? t * n / i : t;
        const int    e = int(floor(    t));
        const int    r = int( ceil(s - t));
        
        sprintf(str, "Processing: %d of %d  Time: %dh %02dm %02ds"
                                    "  Remaining: %dh %02dm %02ds",
                i, n,
                e / 3600, (e % 3600) / 60, (e % 3600) % 60,
                r / 3600, (r % 3600) / 60, (r % 3600) % 60);
    }
    printf("%s", str);
    
    fflush(stdout);
}

/// Write this spherical panorama to the named output file. Use a page size
/// of s and sub-divide to a maximum depth m.

int panoproc::write(const std::string& name,
                    const std::string& text, int n, int m, bool big)
{
    struct timeval t0;
    struct timeval t1;
    
    int pagen = (1 << (2 * m + 3)) - 2;
    int pagei = 0;
    
    int  c = std::min(source->getc(),  3);
    int  b = std::min(source->getb(), 16);
    bool s =          source->gets();
    
    raster   data(     n + 2, n + 2, c, b, s);
    tiff   file(name, text, n + 2, c, b, big);
    pagelist queue;

    // Initialize a breadth-first traversal with the six root cube faces.

    queue.push_back(new page(0, 0, 0, 0, n, 0, 0));
    queue.push_back(new page(0, 0, 1, 0, n, 0, 0));
    queue.push_back(new page(0, 0, 2, 0, n, 0, 0));
    queue.push_back(new page(0, 0, 3, 0, n, 0, 0));
    queue.push_back(new page(0, 0, 4, 0, n, 0, 0));
    queue.push_back(new page(0, 0, 5, 0, n, 0, 0));

    gettimeofday(&t0, 0);
    report(pagen, 0, 0);
    
    // Continue until all pages have been processed.

    while (!queue.empty())
    {
        // Process the next page.

        page *p = queue.front();
        off_t o;

        p->sample(source, &data);

        o = file.append(&data);
        file.link(p->p, o, p->i);

        p->divide(queue, m, o);

        // Dequeue and delete the page.

        queue.pop_front();
        delete p;

        gettimeofday(&t1, 0);
        report(pagen, ++pagei, (t1.tv_sec  - t0.tv_sec) +
                   0.0000001 * (t1.tv_usec - t0.tv_usec));
    }
    printf("\n");
    return 0;
}

//------------------------------------------------------------------------------
