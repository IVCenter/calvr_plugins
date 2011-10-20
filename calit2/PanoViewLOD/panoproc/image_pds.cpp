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

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "image_pds.hpp"
#include "error.hpp"
#include "pds.hpp"
#include "projection.hpp"

//------------------------------------------------------------------------------

image_pds::image_pds() : c(1), b(8)
{
}

image_pds::~image_pds()
{
    std::vector<image_pds_object *>::iterator o;
    
    for (o = objects.begin(); o != objects.end(); ++o)
        delete (*o);
}

//------------------------------------------------------------------------------

static inline int toint(double d)
{
    double c =  ceil(d);
    double f = floor(d);
    return (c - d < d - f) ? int(c) : int(f);
}

static inline double todegrees(double r)
{
    return r * 180.0 / M_PI;
}

static inline double toradians(double d)
{
    return M_PI * d / 180.0;
}

static inline double tolon(double a)
{
    double b = fmod(a, 2 * M_PI);
    return b < 0 ? b + 2 * M_PI : b;
}

//------------------------------------------------------------------------------

// Load the named PDS object. Note the largest channel and bit depth of all
// loaded objects to ensure no loss during export.

void image_pds::source(const std::string& arg)
{
    image_pds_object *o = new image_pds_object(arg);
    
    if (o->data)
    {
        objects.push_back(o);

        c = std::max(c, o->data->getc());
        b = std::max(b, o->data->getb());
    }
}

// Sample the sphere along vector v. Begin by determining the first object to
// include the vector. Note this object, and pass its local position to the
// filter. This will cause several invocations of the get function, below.

void image_pds::get(double *p, const double *v)
{
    const double lat = asin(v[1]), lon = tolon(atan2(v[0], -v[2]));

    int    obj;
    double row;
    double col;

    if ((obj = locate(lat, lon, row, col)) >= 0)
    {
        filter(obj, row, col, p);
    }
    else
    {
        p[0] = 0;
        p[1] = 0;
        p[2] = 0;
        p[3] = 0;
    }
}

// Return the requested pixel of the current object. If this pixel falls outside
// the object, seek a fallback object and sample it directly. While we would
// like to make a filtered sampling of the fallback, this would result in an
// infinite loop of filter tapping across the object boundary.

void image_pds::tap(int obj, int row, int col, double *p) const
{
    if (objects[obj]->get(row, col, p) == false)
    {
        p[0] = 0;
        p[1] = 0;
        p[2] = 0;
        p[3] = 0;

        if (objects[obj]->proj)
        {
            double lat = 0, r;
            double lon = 0, c;
            int    alt;

            objects[obj]->proj->tolatlon(row, col, lat, lon);
        
            if ((alt = locate(lat, lon, r, c)) >= 0)
                objects[alt]->get(toint(r),
                                  toint(c), p);
        }
    }
}

// Determine the first, if any, of the loaded objects to contain the given
// position. Return turn it along with its determination of the pixel position.

int image_pds::locate(double  lat, double  lon,
                      double &row, double &col) const
{
    for (int obj = 0; obj < int(objects.size()); ++obj)

        if (objects[obj]->proj &&
            objects[obj]->proj->iswithin(lat, lon))
        {
            objects[obj]->proj->torowcol(lat, lon, row, col);
            return (obj);
        }
    
    return -1;
}

//------------------------------------------------------------------------------

image_pds_object::image_pds_object(const std::string& name) : proj(0), data(0)
{
    unsigned int H[65536];
    
    load(name);
    
    data->accum_histogram(H, 0);
}

image_pds_object::~image_pds_object()
{
    delete data;
    delete proj;
}

// Get the pixel value at the given position. Return false to indicate that the
// position falls outside the bounds of this object.

bool image_pds_object::get(int row, int col, double *p) const
{
    if (0 <= row && row < rows && 0 <= col && col < cols)
    {
        data->get(row, col, p);
        return true;
    }
    else
        return false;
}

//------------------------------------------------------------------------------

static void *map(const char *name, int& fd, size_t& sz)
{
    struct stat st;
    void        *p;
    
    if ((stat(name, &st)) != -1 && (sz = st.st_size))
    {
        if ((fd = open(name, O_RDONLY)) != -1)
        {
            if ((p = mmap(0, sz, PROT_READ, MAP_PRIVATE, fd, 0)) != MAP_FAILED)
            {
                return p;
            }
            else sys_err_mesg("mmap %s", name);            
        }
        else sys_err_mesg("open %s", name);        
    }
    else sys_err_mesg("stat %s", name);
    
    return 0;
}

// Parse the given PDS data buffer and initialize all data values.

void image_pds_object::load(const std::string& name)
{
    size_t sz = 0;
    int    fd = 0;
    
    if (const char *str = (const char *) map(name.c_str(), fd, sz))
    {
        // Parse the data file name and projection type.

        std::string file_name = pds_string(str, "FILE_NAME");
        std::string proj_type = pds_string(str, "MAP_PROJECTION_TYPE");
        std::string samp_type = pds_string(str, "SAMPLE_TYPE");

        if      (proj_type == "EQUIRECTANGULAR")
            proj = new equirectangular(str);
        else if (proj_type == "ORTHOGRAPHIC")
            proj = new orthographic(str);
        else if (proj_type == "POLAR STEREOGRAPHIC")
            proj = new polar_stereographic(str);
        else if (proj_type == "SIMPLE CYLINDRICAL")
            proj = new simple_cylindrical(str);

        // Extract the raster format.

        cols    =          pds_length(str,  "LINE_SAMPLES");
        rows    =          pds_length(str,         "LINES");    
        int   c = std::max(pds_length(str,         "BANDS"), 1);
        int   b = std::max(pds_length(str,   "SAMPLE_BITS"), 8);
        off_t o =    off_t(pds_length(str,  "RECORD_BYTES")
                         * pds_length(str, "LABEL_RECORDS"));
        bool s = false;

        if (samp_type == "LSB_INTEGER") s = true;
        if (samp_type == "MSB_INTEGER") s = true;
        
        DEBUGPRINT("%d %d %d %d %d %s\n", cols, rows, c, b, s, name.c_str());

        // Instantiate a raster using either an attached or detached data block.

        if (!file_name.empty())
            data = new raster(cols, rows, c, b, s, name, o);
        else
            data = new raster(cols, rows, c, b, s, file_name, 0);

        munmap((void *) str, sz);
    }
    close(fd);
}

//------------------------------------------------------------------------------

//void image_pds_object::hack(const std::string& name)
//{        
//    // This hack limits the range of each of six orthographic projections of the
//    // WAC global mosaic. This dataset is otherwise overspecified times three.
//    
//    if (!search(name, "WAC_GLOBAL_O000X([0-9]+).IMG").empty())
//    {
//        lonmin = tolon(lonp - toradians(30.0));
//        lonmax = tolon(lonp + toradians(30.0));
//        app_wrn_mesg("WAC orthographic HACK applied to %s\n", name.c_str());
//    }
//}

//void image_pds_object::write16(const std::string& name)
//{
//    const int w = data->getw();
//    const int h = data->geth();
//    const int c = data->getc();
//    
//    if (raster *temp = new raster(w, h, c, 16, s))
//    {
//        printf("Generating %s.\n", name.c_str());
//        
//        for     (int i = 0; i < h; ++i)
//            for (int j = 0; j < w; ++j)
//            {
//                double p[3];
//                data->get(i, j, p);
//                temp->put(i, j, p);
//            }
//        
//        printf("Writing %s.\n", name.c_str());
//        
//        if (FILE *fp = fopen(name.c_str(), "wb"))
//        {
//            fwrite(temp->buffer(0), 2, w * h * c, fp);
//            fclose(fp);
//        }
//        
//        delete temp;
//    }
//}

//------------------------------------------------------------------------------
