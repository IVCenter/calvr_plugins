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

#ifndef PANOPROC_IMAGE_DATA_HPP
#define PANOPROC_IMAGE_DATA_HPP

#include "image.hpp"
#include "raster.hpp"

//------------------------------------------------------------------------------

// An image_data object is an image that provides access to stored data.
//
// An image_data allows data sources to be named and implements functions to
// load a variety of data file formats, though the use of these functions is
// defered to the subclasses.
//
// Internally, image_data provides an abstract API to access pixel data, with
// the mechanism defined by subclass. It augments access with linear and cubic
// filtering functions.

class image_data : public image
{
public:

    image_data() : docubic(false) { };

    void setcubic(bool b) { docubic = b; }

    virtual void source(const std::string&) = 0;
    
protected:

    void filter(int, double, double, double *) const;

    virtual void tap(int, int, int, double *) const = 0;
    
    raster *load_tif(const std::string&);
    raster *load_jpg(const std::string&);
    raster *load_png(const std::string&);
    raster *load_img(const std::string&);

private:

    void linear(int, double, double, double *) const;
    void cubic (int, double, double, double *) const;

    bool docubic;
};

//------------------------------------------------------------------------------

#endif
