//  Copyright (C) 2005-2011 Robert Kooima
//
//  THUMB is free software; you can redistribute it and/or modify it under
//  the terms of  the GNU General Public License as  published by the Free
//  Software  Foundation;  either version 2  of the  License,  or (at your
//  option) any later version.
//
//  This program  is distributed in the  hope that it will  be useful, but
//  WITHOUT   ANY  WARRANTY;   without  even   the  implied   warranty  of
//  MERCHANTABILITY  or FITNESS  FOR A  PARTICULAR PURPOSE.   See  the GNU
//  General Public License for more details.

#include <GL/glew.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <limits>

#include "math3d.h"
#include "glyph.h"
#include "glsl.h"
#include "cube.hpp"

#include "sph-model.hpp"

//------------------------------------------------------------------------------

sph_model::sph_model(sph_cache& cache,
                     const char *vert,
                     const char *frag, int n, int d, int s) :
    cache(cache), depth(d), size(s), time(1), status(cube_size(d), s_halt)
{
    init_program(vert, frag);
    init_arrays(n);
    
    zoomv[0] =  0;
    zoomv[1] =  0;
    zoomv[2] = -1;
    zoomk    =  1;
}

sph_model::~sph_model()
{
    free_arrays();
    free_program();
}

//------------------------------------------------------------------------------

static inline double scale(double k, double t)
{
    if (k < 1.0)
        return std::min(t / k, 1.0 - (1.0 - t) * k);
    else
        return std::max(t / k, 1.0 - (1.0 - t) * k);
}

void sph_model::zoom(double *w, const double *v)
{
    double d = vdot(v, zoomv);
    
    if (-1 < d && d < 1)
    {
        double b = scale(zoomk, acos(d) / M_PI) * M_PI;
                
        double y[3];
        double x[3];
        
        vcrs(y, v, zoomv);
        vnormalize(y, y);
        vcrs(x, zoomv, y);
        vnormalize(x, x);
        
        vmul(w, zoomv, cos(b));
        vmad(w, w,  x, sin(b));
    }
    else vcpy(w, v);
}

//------------------------------------------------------------------------------

static void bislerp(double *p, const double *a, const double *b,
                               const double *c, const double *d,
                               double x, double y)
{
    double t[3];
    double u[3];
    
    vslerp(t, a, b, x);
    vslerp(u, c, d, x);
    vslerp(p, t, u, y);

    vnormalize(p, p);
}

static inline double max(double a, double b, double c, double d)
{
    double x = (a > b) ? a : b;
    double y = (c > d) ? c : d;
    return     (x > y) ? x : y;
}

static inline double length(const double *a, const double *b, int w, int h)
{
    if (a[3] <= 0 && b[3] <= 0) return 0;
    if (a[3] <= 0)              return HUGE_VAL;
    if (b[3] <= 0)              return HUGE_VAL;

    double dx = (a[0] / a[3] - b[0] / b[3]) * w / 2;
    double dy = (a[1] / a[3] - b[1] / b[3]) * h / 2;
        
    return sqrt(dx * dx + dy * dy);
}

//------------------------------------------------------------------------------

double sph_model::view_face(const double *M, int vw, int vh,
                            double ee, double ww, double nn, double ss, int j)
{
    double ne[3], a[3], e[3], A[4], E[4];    // North-east corner
    double nw[3], b[3], f[3], B[4], F[4];    // North-west corner
    double se[3], c[3], g[3], C[4], G[4];    // South-east corner
    double sw[3], d[3], h[3], D[4], H[4];    // South-west corner
    
    vnormalize(ne, cube_v[cube_i[j][0]]);
    vnormalize(nw, cube_v[cube_i[j][1]]);
    vnormalize(se, cube_v[cube_i[j][2]]);
    vnormalize(sw, cube_v[cube_i[j][3]]);
    
    bislerp(a, ne, nw, se, sw, ee, nn);
    bislerp(b, ne, nw, se, sw, ww, nn);
    bislerp(c, ne, nw, se, sw, ee, ss);
    bislerp(d, ne, nw, se, sw, ww, ss);
    
    zoom(a, a);
    zoom(b, b);
    zoom(c, c);
    zoom(d, d);
    
    // Compute the maximum extent due to bulge.

    double v[3];
    
    v[0] = a[0] + b[0] + c[0] + d[0];
    v[1] = a[1] + b[1] + c[1] + d[1];
    v[2] = a[2] + b[2] + c[2] + d[2];
    
    double k = vlen(v) / vdot(a, v);

    // Compute the outer corners of the bulge bound.
    
    vmul(e, a, k);
    vmul(f, b, k);
    vmul(g, c, k);
    vmul(h, d, k);

    // Compute W and reject any volume on the far side of the singularity.
    
    A[3] = M[ 3] * a[0] + M[ 7] * a[1] + M[11] * a[2] + M[15];
    B[3] = M[ 3] * b[0] + M[ 7] * b[1] + M[11] * b[2] + M[15];
    C[3] = M[ 3] * c[0] + M[ 7] * c[1] + M[11] * c[2] + M[15];
    D[3] = M[ 3] * d[0] + M[ 7] * d[1] + M[11] * d[2] + M[15];
    E[3] = M[ 3] * e[0] + M[ 7] * e[1] + M[11] * e[2] + M[15];
    F[3] = M[ 3] * f[0] + M[ 7] * f[1] + M[11] * f[2] + M[15];
    G[3] = M[ 3] * g[0] + M[ 7] * g[1] + M[11] * g[2] + M[15];
    H[3] = M[ 3] * h[0] + M[ 7] * h[1] + M[11] * h[2] + M[15];

    if (A[3] <= 0 && B[3] <= 0 && C[3] <= 0 && D[3] <= 0 &&
        E[3] <= 0 && F[3] <= 0 && G[3] <= 0 && H[3] <= 0)
        return 0;

    // Compute Z and apply the near and far clipping planes.

    A[2] = M[ 2] * a[0] + M[ 6] * a[1] + M[10] * a[2] + M[14];
    B[2] = M[ 2] * b[0] + M[ 6] * b[1] + M[10] * b[2] + M[14];
    C[2] = M[ 2] * c[0] + M[ 6] * c[1] + M[10] * c[2] + M[14];
    D[2] = M[ 2] * d[0] + M[ 6] * d[1] + M[10] * d[2] + M[14];
    E[2] = M[ 2] * e[0] + M[ 6] * e[1] + M[10] * e[2] + M[14];
    F[2] = M[ 2] * f[0] + M[ 6] * f[1] + M[10] * f[2] + M[14];
    G[2] = M[ 2] * g[0] + M[ 6] * g[1] + M[10] * g[2] + M[14];
    H[2] = M[ 2] * h[0] + M[ 6] * h[1] + M[10] * h[2] + M[14];
    
    if (A[2] >  A[3] && B[2] >  B[3] && C[2] >  C[3] && D[2] >  D[3] &&
        E[2] >  E[3] && F[2] >  F[3] && G[2] >  G[3] && H[2] >  H[3])
        return 0;
    if (A[2] < -A[3] && B[2] < -B[3] && C[2] < -C[3] && D[2] < -D[3] &&
        E[2] < -E[3] && F[2] < -F[3] && G[2] < -G[3] && H[2] < -H[3])
        return 0;

    // Compute Y and apply the bottom and top clipping planes.

    A[1] = M[ 1] * a[0] + M[ 5] * a[1] + M[ 9] * a[2] + M[13];
    B[1] = M[ 1] * b[0] + M[ 5] * b[1] + M[ 9] * b[2] + M[13];
    C[1] = M[ 1] * c[0] + M[ 5] * c[1] + M[ 9] * c[2] + M[13];
    D[1] = M[ 1] * d[0] + M[ 5] * d[1] + M[ 9] * d[2] + M[13];
    E[1] = M[ 1] * e[0] + M[ 5] * e[1] + M[ 9] * e[2] + M[13];
    F[1] = M[ 1] * f[0] + M[ 5] * f[1] + M[ 9] * f[2] + M[13];
    G[1] = M[ 1] * g[0] + M[ 5] * g[1] + M[ 9] * g[2] + M[13];
    H[1] = M[ 1] * h[0] + M[ 5] * h[1] + M[ 9] * h[2] + M[13];
    
    if (A[1] >  A[3] && B[1] >  B[3] && C[1] >  C[3] && D[1] >  D[3] &&
        E[1] >  E[3] && F[1] >  F[3] && G[1] >  G[3] && H[1] >  H[3])
        return 0;
    if (A[1] < -A[3] && B[1] < -B[3] && C[1] < -C[3] && D[1] < -D[3] &&
        E[1] < -E[3] && F[1] < -F[3] && G[1] < -G[3] && H[1] < -H[3])
        return 0;

    // Compute X and apply the left and right clipping planes.

    A[0] = M[ 0] * a[0] + M[ 4] * a[1] + M[ 8] * a[2] + M[12];
    B[0] = M[ 0] * b[0] + M[ 4] * b[1] + M[ 8] * b[2] + M[12];
    C[0] = M[ 0] * c[0] + M[ 4] * c[1] + M[ 8] * c[2] + M[12];
    D[0] = M[ 0] * d[0] + M[ 4] * d[1] + M[ 8] * d[2] + M[12];
    E[0] = M[ 0] * e[0] + M[ 4] * e[1] + M[ 8] * e[2] + M[12];
    F[0] = M[ 0] * f[0] + M[ 4] * f[1] + M[ 8] * f[2] + M[12];
    G[0] = M[ 0] * g[0] + M[ 4] * g[1] + M[ 8] * g[2] + M[12];
    H[0] = M[ 0] * h[0] + M[ 4] * h[1] + M[ 8] * h[2] + M[12];
    
    if (A[0] >  A[3] && B[0] >  B[3] && C[0] >  C[3] && D[0] >  D[3] &&
        E[0] >  E[3] && F[0] >  F[3] && G[0] >  G[3] && H[0] >  H[3])
        return 0;
    if (A[0] < -A[3] && B[0] < -B[3] && C[0] < -C[3] && D[0] < -D[3] &&
        E[0] < -E[3] && F[0] < -F[3] && G[0] < -G[3] && H[0] < -H[3])
        return 0;

    // Compute the length of the longest visible edge, in pixels.

//  return max(length(A, B, vw, vh),
//             length(C, D, vw, vh),
//             length(A, C, vw, vh),
//             length(B, D, vw, vh));
    return std::max(max(length(A, B, vw, vh),
                        length(C, D, vw, vh),
                        length(A, C, vw, vh),
                        length(B, D, vw, vh)),
                    max(length(E, F, vw, vh),
                        length(G, H, vw, vh),
                        length(E, G, vw, vh),
                        length(F, H, vw, vh)));
}

//------------------------------------------------------------------------------

void sph_model::prep(const double *P, const double *V, int w, int h)
{
    double M[16];
    
    mmultiply(M, P, V);
        
    for (int i = 0; i < 6; ++i)
        prep_face(M, w, h, 0, 1, 0, 1, i, 0, i);
}

void sph_model::prep_face(const double *M, int w, int h,
                          double r, double l,
                          double t, double b, int j, int d, int i)

{
    if (d < depth)
    {
        double s = view_face(M, w, h, r, l, t, b, j);

        if (s > 0)
        {
            const int i0 = face_child(i, 0);
            const int i1 = face_child(i, 1);
            const int i2 = face_child(i, 2);
            const int i3 = face_child(i, 3);

            if (s < size)
            {
                status[i]  = s_draw;
                status[i0] = s_halt;
                status[i1] = s_halt;
                status[i2] = s_halt;
                status[i3] = s_halt;
            }
            else
            {
                const double x = (r + l) * 0.5;
                const double y = (t + b) * 0.5;

                prep_face(M, w, h, r, x, t, y, j, d + 1, i0);
                prep_face(M, w, h, x, l, t, y, j, d + 1, i1);
                prep_face(M, w, h, r, x, y, b, j, d + 1, i2);
                prep_face(M, w, h, x, l, y, b, j, d + 1, i3);

                if (status[i0] == s_halt &&
                    status[i1] == s_halt &&
                    status[i2] == s_halt &&
                    status[i3] == s_halt)

                    status[i] = s_halt;
                else
                    status[i] = s_pass;
            }
        }
        else
            status[i] = s_halt;
    }
    else
    {
        double s = view_face(M, w, h, r, l, t, b, j);

        if (s > 0)
            status[i] = s_draw;
        else
            status[i] = s_halt;
    }
}

//------------------------------------------------------------------------------

GLfloat sph_model::age(int then)
{
    GLfloat a = GLfloat(time - then) / 60.f;
    return (a > 1.f) ? 1.f : a;
}

void sph_model::draw(const double *P, const double *V, const int *fv, int fc,
                                                       const int *pv, int pc)
{
    double M[16];
    
    mmultiply(M, P, V);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(P);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(V);
    
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, 0);

    for (int i = 15; i >= 0; --i)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, cache.get_fill());
    }

    // This is a hack that ensures that the root pages of all files are touched.

    GLuint o;
    int tock;

    for (int i = 0; i < fc; ++i)
    {
        o = cache.get_page(fv[i], 0, time, tock);
        o = cache.get_page(fv[i], 1, time, tock);
        o = cache.get_page(fv[i], 2, time, tock);
        o = cache.get_page(fv[i], 3, time, tock);
        o = cache.get_page(fv[i], 4, time, tock);
        o = cache.get_page(fv[i], 5, time, tock);
    }
    for (int i = 0; i < pc; ++i)
    {
        o = cache.get_page(pv[i], 0, time, tock);
        o = cache.get_page(pv[i], 1, time, tock);
        o = cache.get_page(pv[i], 2, time, tock);
        o = cache.get_page(pv[i], 3, time, tock);
        o = cache.get_page(pv[i], 4, time, tock);
        o = cache.get_page(pv[i], 5, time, tock);
    }

#if 0
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
#endif

    glUseProgram(program);
    {
        glUniform1f(glGetUniformLocation(program, "zoomk"), zoomk);
        glUniform3f(glGetUniformLocation(program, "zoomv"),
                    zoomv[0], zoomv[1], zoomv[2]);

        for (int i = 0; i < 6; ++i)
        {
            double a[3], b[3], c[3], d[3];
            
            vnormalize(a, cube_v[cube_i[i][0]]);
            vnormalize(b, cube_v[cube_i[i][1]]);
            vnormalize(c, cube_v[cube_i[i][2]]);
            vnormalize(d, cube_v[cube_i[i][3]]);
            
            glUniform3f(pos_a, GLfloat(a[0]), GLfloat(a[1]), GLfloat(a[2]));
            glUniform3f(pos_b, GLfloat(b[0]), GLfloat(b[1]), GLfloat(b[2]));
            glUniform3f(pos_c, GLfloat(c[0]), GLfloat(c[1]), GLfloat(c[2]));
            glUniform3f(pos_d, GLfloat(d[0]), GLfloat(d[1]), GLfloat(d[2]));

            draw_face(fv, fc, pv, pc, 0, 1, 0, 1, 0, i);
        }
    }
    glUseProgram(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER,         0);
    glDisableClientState(GL_VERTEX_ARRAY);

    glActiveTexture(GL_TEXTURE0);
}

void sph_model::draw_face(const int *fv, int fc,
                          const int *pv, int pc,
                          double r, double l, double t, double b, int d, int i)
{
    GLuint o = 0;
    int then = time;

    if (status[i] != s_halt)
    {
        for (int fi = 0; fi < fc; ++fi)
        {
            int e = fi * 8 + d;
            
            glActiveTexture(GL_TEXTURE0 + e);

            o = cache.get_page(fv[fi], i, time, then);

            glUniform1f(alpha[e], age(then));
            glBindTexture(GL_TEXTURE_2D, o);
        }
        glUniform2f(tex_a[d], GLfloat(r), GLfloat(t));
        glUniform2f(tex_d[d], GLfloat(l), GLfloat(b));
    }

    if (status[i] == s_pass)
    {
        for (int pi = 0; pi < pc; ++pi)
            cache.get_page(pv[pi], i, time, then);
    }
    
    if (status[i] == s_pass)
    {
        const double x = (r + l) * 0.5;
        const double y = (t + b) * 0.5;

        draw_face(fv, fc, pv, pc, r, x, t, y, d + 1, face_child(i, 0));
        draw_face(fv, fc, pv, pc, x, l, t, y, d + 1, face_child(i, 1));
        draw_face(fv, fc, pv, pc, r, x, y, b, d + 1, face_child(i, 2));
        draw_face(fv, fc, pv, pc, x, l, y, b, d + 1, face_child(i, 3));
    }

    if (status[i] == s_draw)
    {
        int n, s, e, w, j = 0;
        
        face_neighbors(i, n, s, e, w);
        
        if (i > 5) j = (status[face_parent(n)] == s_draw ? 1 : 0)
                     | (status[face_parent(s)] == s_draw ? 2 : 0)
                     | (status[face_parent(e)] == s_draw ? 4 : 0)
                     | (status[face_parent(w)] == s_draw ? 8 : 0);

        glUniform1i(level, d);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elements[j]);
        glDrawElements(GL_QUADS, count, GL_UNSIGNED_SHORT, 0);
    }
    
    if (status[i] != s_halt)
    {
        for (int fi = 0; fi < fc; ++fi)
        {
            glActiveTexture(GL_TEXTURE0 + fi * 8 + d);
            glBindTexture(GL_TEXTURE_2D, cache.get_fill());
        }
    }
}

//------------------------------------------------------------------------------

void sph_model::set_fade(double k)
{
    glUseProgram(program);
    glUniform1f(fader, k);
}

static GLuint glGetUniformLocationv(GLuint program, const char *fmt, int d)
{
    char str[256];
    
    sprintf(str, fmt, d);
    
    return glGetUniformLocation(program, str);
}

void sph_model::init_program(const char *vert_src,
                             const char *frag_src)
{    
    if (vert_src && frag_src)
    {
        vert_shader = glsl_init_shader(GL_VERTEX_SHADER,   vert_src);
        frag_shader = glsl_init_shader(GL_FRAGMENT_SHADER, frag_src);
        program     = glsl_init_program(vert_shader, frag_shader);

        glUseProgram(program);
        
        pos_a = glGetUniformLocation(program, "pos_a");
        pos_b = glGetUniformLocation(program, "pos_b");
        pos_c = glGetUniformLocation(program, "pos_c");
        pos_d = glGetUniformLocation(program, "pos_d");
        level = glGetUniformLocation(program, "level");
        fader = glGetUniformLocation(program, "fader");

        for (int d = 0; d < 8; ++d)
        {
            tex_a[d]  = glGetUniformLocationv(program, "tex_a[%d]", d);
            tex_d[d]  = glGetUniformLocationv(program, "tex_d[%d]", d);
        }

        for (int d = 0; d < 16; ++d)
        {
            alpha[d]  = glGetUniformLocationv(program, "alpha[%d]", d);
            glUniform1i(glGetUniformLocationv(program, "image[%d]", d), d);
        }
    }
}

void sph_model::free_program()
{
    glDeleteProgram(program);
    glDeleteShader(frag_shader);
    glDeleteShader(vert_shader);
}

//------------------------------------------------------------------------------

static inline int up(int d)
{
    return (d % 2) ? d + 1 : d;
}

static inline int dn(int d)
{
    return (d % 2) ? d - 1 : d;
}

static void init_vertices(int n)
{
    struct vertex
    {
        GLfloat x;
        GLfloat y;
    };
    
    const size_t s = (n + 1) * (n + 1) * sizeof (vertex);
    
    if (vertex *p = (vertex *) malloc(s))
    {
        vertex *v = p;
        
        // Compute the position of each vertex.
        
        for     (int r = 0; r <= n; ++r)
            for (int c = 0; c <= n; ++c, ++v)
            {
                v->x = GLfloat(c) / GLfloat(n);
                v->y = GLfloat(r) / GLfloat(n);
            }
        
        // Upload the vertices to the vertex buffer.
        
        glBufferData(GL_ARRAY_BUFFER, s, p, GL_STATIC_DRAW); 
        free(p);
    }
}

static void init_elements(int n, int b)
{
    struct element
    {
        GLshort a;
        GLshort b;
        GLshort d;
        GLshort c;
    };
    
    const size_t s = n * n * sizeof (element);
    const int    d = n + 1;
    
    if (element *p = (element *) malloc(s))
    {
        element *e = p;
        
        // Compute the indices for each quad.
        
        for     (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c, ++e)
            {
                e->a = GLshort(d * (r    ) + (c    ));
                e->b = GLshort(d * (r    ) + (c + 1));
                e->c = GLshort(d * (r + 1) + (c    ));
                e->d = GLshort(d * (r + 1) + (c + 1));
            }

        // Rewind the indices to reduce edge resolution as necessary.

        element *N = p;
        element *W = p + (n - 1);
        element *E = p;
        element *S = p + (n - 1) * n;

        for (int i = 0; i < n; ++i, N += 1, S += 1, E += n, W += n)
        {
            if (b & 1) { if (i & 1) N->a -= 1; else N->b -= 1; }
            if (b & 2) { if (i & 1) S->c += 1; else S->d += 1; }
            if (b & 4) { if (i & 1) E->a += d; else E->c += d; }
            if (b & 8) { if (i & 1) W->b -= d; else W->d -= d; }
        }
        
        // Upload the indices to the element buffer.
        
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, s, p, GL_STATIC_DRAW);
        free(p);
    }
}

void sph_model::init_arrays(int n)
{
    glGenBuffers(1, &vertices);
    glGenBuffers(16, elements);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    init_vertices(n);
    
    for (int b = 0; b < 16; ++b)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elements[b]);
        init_elements(n, b);
    }
    
    count = 4 * n * n;
}

void sph_model::free_arrays()
{
    glDeleteBuffers(16, elements);
    glDeleteBuffers(1, &vertices);
}

//------------------------------------------------------------------------------
