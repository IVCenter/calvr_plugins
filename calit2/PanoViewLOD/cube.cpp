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

#include "cube.hpp"

//-----------------------------------------------------------------------------

const double cube_v[8][3] = {
    {  1,  1,  1 },
    { -1,  1,  1 },
    {  1, -1,  1 },
    { -1, -1,  1 },
    {  1,  1, -1 },
    { -1,  1, -1 },
    {  1, -1, -1 },
    { -1, -1, -1 },
};

const int cube_i[6][4] = {
    { 0, 4, 2, 6 },
    { 5, 1, 7, 3 },
    { 5, 4, 1, 0 },
    { 3, 2, 7, 6 },
    { 1, 0, 3, 2 },
    { 4, 5, 6, 7 },
};

// Calculate the integer binary log of n ---------------------------------------

int log2(int n)
{
    int r = 0;
    
    if (n >= (1 << 16)) { n >>= 16; r += 16; }
    if (n >= (1 <<  8)) { n >>=  8; r +=  8; }
    if (n >= (1 <<  4)) { n >>=  4; r +=  4; }
    if (n >= (1 <<  2)) { n >>=  2; r +=  2; }
    if (n >= (1 <<  1)) {           r +=  1; }

    return r;
}

// Calculate the number of faces in a cube with depth n. -----------------------

int cube_size(int n)
{
    return (1 << (2 * n + 3)) - 2;
}

// Calculate the recursion level at which face i appears. ----------------------

int face_level(int i)
{
    return (log2(i + 2) - 1) / 2;
}

// Calculate the parent index of face i. Assume i > 5. -------------------------

int face_parent(int i)
{
    return (i - 6) / 4;
}

// Calculate child i of face p. ------------------------------------------------

int face_child(int p, int i)
{
    return 6 + 4 * p + i;
}

// Calculate the child index of face i. Assume i > 5. --------------------------

int face_index(int i)
{
    return (i - 6) - ((i - 6) / 4) * 4;
}

// Find the index of face (i, j) of (s, s) in the tree rooted at face p. -------

int face_locate(int p, int i, int j, int s)
{
    if (s > 1)
    {
        int c = 0;
        
        s >>= 1;
        
        if (i >= s) c |= 2;
        if (j >= s) c |= 1;
        
        return face_locate(face_child(p, c), i % s, j % s, s);
    }
    else return p;
}

// Find the indices of the four neighbors of face p. ---------------------------

void face_neighbors(int p, int& u, int& d, int& r, int& l)
{
    struct turn
    {
        unsigned int p : 3;
        unsigned int r : 3;
        unsigned int c : 3;
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

    int o[6];
    int i = 0;
    int j = 0;
    int s;
    
    for (s = 1; p > 5; s <<= 1, p = face_parent(p))
    {
        if (face_index(p) & 1) j += s;
        if (face_index(p) & 2) i += s;
    }

    o[0] = 0;
    o[1] = i;
    o[2] = j;
    o[3] = s     - 1;
    o[4] = s - i - 1;
    o[5] = s - j - 1;

    if (i     != 0) u = face_locate(p, i - 1, j, s);
    else            u = face_locate(uu[p].p, o[uu[p].r], o[uu[p].c], s);

    if (i + 1 != s) d = face_locate(p, i + 1, j, s);
    else            d = face_locate(dd[p].p, o[dd[p].r], o[dd[p].c], s);

    if (j     != 0) r = face_locate(p, i, j - 1, s);
    else            r = face_locate(rr[p].p, o[rr[p].r], o[rr[p].c], s);

    if (j + 1 != s) l = face_locate(p, i, j + 1, s);
    else            l = face_locate(ll[p].p, o[ll[p].r], o[ll[p].c], s);
}

//------------------------------------------------------------------------------
