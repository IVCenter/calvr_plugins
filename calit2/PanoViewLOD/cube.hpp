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

#ifndef PANOVIEW_CUBE_HPP
#define PANOVIEW_CUBE_HPP

//-----------------------------------------------------------------------------

int log2(int n);

//-----------------------------------------------------------------------------

extern const double cube_v[8][3];
extern const int    cube_i[6][4];

int  cube_size(int);

int  face_locate(int, int, int, int);
int  face_child (int, int);
int  face_index (int);
int  face_level (int);
int  face_parent(int);

void face_neighbors(int, int&, int&, int&, int&);

//-----------------------------------------------------------------------------

#endif
