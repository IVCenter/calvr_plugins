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

#ifndef PANOPROC_IMAGE_BASIC_HPP
#define PANOPROC_IMAGE_BASIC_HPP

#include <vector>

#include "image_data.hpp"
#include "raster.hpp"

//------------------------------------------------------------------------------

// An image_basic object represents a basic tiled raster image_data. It provides
// an implementation of the source method that loads and stores each named data
// file in a raster. It also provides a data get that samples the appropriate
// raster and converts global pixel coordinates to local.

class image_basic : public image_data
{
public:

    image_basic(const char *);
   ~image_basic();

    virtual int  getc() const { return c;     }
    virtual int  getb() const { return b;     }
    virtual bool gets() const { return false; }

    virtual void source(const std::string&);
    
protected:

    virtual void tap(int, int, int, double *) const;
    
    int w;
    int h;
    int c;
    int b;
    int R;
    int C;
    
private:

    std::vector<raster *> data;
};

//------------------------------------------------------------------------------

#endif
