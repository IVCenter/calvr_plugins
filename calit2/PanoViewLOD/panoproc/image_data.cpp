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

#include <cmath>

#include "image_data.hpp"
#include "error.hpp"

//------------------------------------------------------------------------------

static inline double lerp(double a, double b, double t)
{
    return b * t + a * (1 - t);
}

static inline double cerp(double a, double b, double c, double d, double t)
{
    return b + (-a / 2                 + c / 2        ) * t
             + ( a     - 5 * b / 2 + 2 * c     - d / 2) * t * t
             + (-a / 2 + 3 * b / 2 - 3 * c / 2 + d / 2) * t * t * t;
}

//------------------------------------------------------------------------------

void image_data::filter(int d, double i, double j, double *p) const
{
    if (docubic)
        cubic (d, i, j, p);
    else
        linear(d, i, j, p);
}

void image_data::linear(int d, double i, double j, double *p) const
{
    const double s = i - floor(i);
    const double t = j - floor(j);

    const int ia = int(floor(i));
    const int ib = int( ceil(i));
    const int ja = int(floor(j));
    const int jb = int( ceil(j));

    double aa[3];
    double ab[3];
    double ba[3];
    double bb[3];

    tap(d, ia, ja, aa);
    tap(d, ia, jb, ab);
    tap(d, ib, ja, ba);
    tap(d, ib, jb, bb);

    switch (getc())
    {
        case 3: p[2] = lerp(lerp(aa[2], ab[2], t), lerp(ba[2], bb[2], t), s);
        case 2: p[1] = lerp(lerp(aa[1], ab[1], t), lerp(ba[1], bb[1], t), s);
        case 1: p[0] = lerp(lerp(aa[0], ab[0], t), lerp(ba[0], bb[0], t), s);
    }
}

void image_data::cubic(int d, double i, double j, double *p) const
{
    const double s = i - floor(i);
    const double t = j - floor(j);

    const int ib = int(floor(i));
    const int ic = int( ceil(i));
    const int jb = int(floor(j));
    const int jc = int( ceil(j));

    const int ia = ib - 1;
    const int id = ic + 1;
    const int ja = jb - 1;
    const int jd = jc + 1;
    
    double aa[3], ab[3], ac[3], ad[3];
    double ba[3], bb[3], bc[3], bd[3];
    double ca[3], cb[3], cc[3], cd[3];
    double da[3], db[3], dc[3], dd[3];
    
    tap(d, ia, ja, aa);
    tap(d, ia, jb, ab);
    tap(d, ia, jc, ac);
    tap(d, ia, jd, ad);
    tap(d, ib, ja, ba);
    tap(d, ib, jb, bb);
    tap(d, ib, jc, bc);
    tap(d, ib, jd, bd);
    tap(d, ic, ja, ca);
    tap(d, ic, jb, cb);
    tap(d, ic, jc, cc);
    tap(d, ic, jd, cd);
    tap(d, id, ja, da);
    tap(d, id, jb, db);
    tap(d, id, jc, dc);
    tap(d, id, jd, dd);
    
    switch (getc())
    {
        case 3: p[2] = cerp(cerp(aa[2], ab[2], ac[2], ad[2], t),
                            cerp(ba[2], bb[2], bc[2], bd[2], t),
                            cerp(ca[2], cb[2], cc[2], cd[2], t),
                            cerp(da[2], db[2], dc[2], dd[2], t), s);
        case 2: p[1] = cerp(cerp(aa[1], ab[1], ac[1], ad[1], t),
                            cerp(ba[1], bb[1], bc[1], bd[1], t),
                            cerp(ca[1], cb[1], cc[1], cd[1], t),
                            cerp(da[1], db[1], dc[1], dd[1], t), s);
        case 1: p[0] = cerp(cerp(aa[0], ab[0], ac[0], ad[0], t),
                            cerp(ba[0], bb[0], bc[0], bd[0], t),
                            cerp(ca[0], cb[0], cc[0], cd[0], t),
                            cerp(da[0], db[0], dc[0], dd[0], t), s);
    }
}

//------------------------------------------------------------------------------

#include <tiffio.h>

raster *image_data::load_tif(const std::string& name)
{
    raster *d = 0;
    TIFF   *T = 0;

    TIFFSetWarningHandler(0);

    if ((T = TIFFOpen(name.c_str(), "r")))
    {
        uint32 W, H;
        uint16 B, C;

        TIFFGetField(T, TIFFTAG_IMAGEWIDTH,      &W);
        TIFFGetField(T, TIFFTAG_IMAGELENGTH,     &H);
        TIFFGetField(T, TIFFTAG_BITSPERSAMPLE,   &B);
        TIFFGetField(T, TIFFTAG_SAMPLESPERPIXEL, &C);

        if ((d = new raster(int(W), int(H), int(C), int(B), false)))

            for (int i = 0; i < d->geth(); ++i)
                TIFFReadScanline(T, d->buffer(i), i, 0);

        TIFFClose(T);
    }
    return d;
}

//------------------------------------------------------------------------------

#include <jpeglib.h>

raster *image_data::load_jpg(const std::string& name)
{
    raster *d = 0;
    FILE  *fp = 0;

    if ((fp = fopen(name.c_str(), "rb")))
    {
        struct jpeg_decompress_struct cinfo;
        struct jpeg_error_mgr         jerr;

        cinfo.err = jpeg_std_error(&jerr);

        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src        (&cinfo, fp);
        jpeg_read_header      (&cinfo, TRUE);
        jpeg_start_decompress (&cinfo);
        
        if ((d = new raster(int(cinfo.output_width), 
                            int(cinfo.output_height),
                            int(cinfo.output_components), 8, false)))

            while (cinfo.output_scanline < cinfo.output_height)
            {
                JSAMPLE *p = (JSAMPLE *) d->buffer(cinfo.output_scanline);
                jpeg_read_scanlines(&cinfo, &p, 1);
            }

        jpeg_finish_decompress (&cinfo);
        jpeg_destroy_decompress(&cinfo);

        fclose(fp);
    }
    return d;
}

//------------------------------------------------------------------------------

#include <png.h>

raster *image_data::load_png(const std::string& name)
{
    png_structp rp = 0;
    png_infop   ip = 0;
    FILE       *fp = 0;
    raster      *d = 0;

    if ((fp = fopen(name.c_str(), "rb")))
    {
        if ((rp = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0)) &&
            (ip = png_create_info_struct(rp)))
        {
            if (setjmp(png_jmpbuf(rp)) == 0)
            {
                png_init_io  (rp, fp);
                png_read_info(rp, ip);
                png_set_swap (rp);
                
                if (png_get_interlace_type(rp, ip) == PNG_INTERLACE_NONE)
                {
                    if ((d = new raster(int(png_get_image_width (rp, ip)),
                                        int(png_get_image_height(rp, ip)),
                                        int(png_get_channels    (rp, ip)),
                                        int(png_get_bit_depth   (rp, ip)),
                                        false)))

                        for (int i = 0; i < d->geth(); ++i)
                            png_read_row(rp, (png_bytep) d->buffer(i), 0);
                }
                else app_err_exit("%s interlace not supported.", name.c_str());
            }
            png_destroy_read_struct(&rp, &ip, NULL);
        } 
        fclose(fp);
    }
    return d;
}

//------------------------------------------------------------------------------
