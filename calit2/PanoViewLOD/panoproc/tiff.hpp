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

#ifndef PANOPROC_TIFF_HPP
#define PANOPROC_TIFF_HPP

#include <cstdio>
#include <string>

#include "raster.hpp"

//------------------------------------------------------------------------------

class tiff
{
public:

    tiff(const std::string&,
         const std::string&, size_t, size_t, size_t, bool);
   ~tiff();

    off_t append(raster *);
    bool  read  (raster *);

    void link(off_t, off_t, int);
    void next(off_t, off_t);

private:

    template <typename ifd> bool get_ifd(off_t, ifd&);
    template <typename ifd> bool put_ifd(off_t, ifd&);

    void ffd();
    void pad();

    FILE  *file;   // File pointer
    off_t  prev;   // File offset of previous append
    void  *buff;   // Compression buffer pointer
    size_t text;   // Copyright text size
    bool   big;    // Is BigTIFF?
};

//------------------------------------------------------------------------------

#endif