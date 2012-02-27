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

#ifndef PANOPROC_IMAGE_PDS_HPP
#define PANOPROC_IMAGE_PDS_HPP

#include <string>
#include <vector>

#include "image_data.hpp"
#include "projection.hpp"
#include "raster.hpp"

//------------------------------------------------------------------------------

// An image_pds represents an image_data capable of directly interpreting and
// merging data sets conforming to the NASA Planetary Data System standards.
// It accepts named data sources in attached (IMG) and detached (IMG + LBL)
// PDS form. It implements a data sample function that selects the first PDS
// object named that covers the requested datum.

class image_pds_object;

class image_pds : public image_data
{
public:

    image_pds();
   ~image_pds();

    virtual int  getc() const { return c; }
    virtual int  getb() const { return b; }
    virtual bool gets() const { return s; }

    virtual void source(const std::string&);

    virtual void get(double *, const double *);    

private:

    int  c;
    int  b;
    bool s;

    std::vector<image_pds_object *> objects;

    int locate(double, double, double&, double&) const;
    
    virtual void tap(int, int, int, double *) const;
};

//------------------------------------------------------------------------------

class image_pds_object
{
public:

    image_pds_object(const std::string&);
   ~image_pds_object();

    bool get(int, int, double *) const;

    friend class image_pds;

private:

    void load(const std::string&);
    void hack(const std::string&);

    projection *proj;
    raster     *data;

    int cols;
    int rows;
};

//------------------------------------------------------------------------------

#endif
