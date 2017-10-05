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

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdint.h>
#include <zlib.h>
#include <stddef.h>

#include "tiff.hpp"

//------------------------------------------------------------------------------

#pragma pack(push)
#pragma pack(2)

struct header32
{
    uint16_t endianness;
    uint16_t version;
    uint32_t first_ifd;
    
    void init(size_t);
};

struct header64
{
    uint16_t endianness;
    uint16_t version;
    uint16_t offsetsize;
    uint16_t zero;
    uint64_t first_ifd;

    void init(size_t);
};

//------------------------------------------------------------------------------

struct field32
{
    uint16_t tag;
    uint16_t type;
    uint32_t count;
    uint32_t offset;

    void init(int, int, size_t, off_t);
};

struct field64
{
    uint16_t tag;
    uint16_t type;
    uint64_t count;
    uint64_t offset;

    void init(int, int, size_t, off_t);
};

//------------------------------------------------------------------------------

struct ifd32
{
    uint16_t count;
    
    field32  subfile_type;
    field32  image_width;
    field32  image_length;
    field32  bits_per_sample;
    field32  compression;
    field32  interpretation;
    field32  strip_offsets;
    field32  orientation;
    field32  samples_per_pixel;
    field32  strip_byte_counts;
    field32  configuration;
    field32  datetime;
    field32  predictor;
    field32  sub_ifds;
    field32  copyright;
    
    uint32_t next;

    uint16_t bps_value[4];
    uint32_t ifd_value[4];
    char     date_str[20];

    void init(off_t, size_t, size_t, size_t, size_t, size_t);
};

struct ifd64
{
    uint64_t count;
    
    field64  subfile_type;
    field64  image_width;
    field64  image_length;
    field64  bits_per_sample;
    field64  compression;
    field64  interpretation;
    field64  strip_offsets;
    field64  orientation;
    field64  samples_per_pixel;
    field64  strip_byte_counts;
    field64  configuration;
    field64  datetime;
    field64  predictor;
    field64  sub_ifds;
    field64  copyright;
    
    uint64_t next;

    uint16_t bps_value[4];
    uint64_t ifd_value[4];
    char     date_str[20];
    
    void init(off_t, size_t, size_t, size_t, size_t, size_t);
};

#pragma pack(pop)

//------------------------------------------------------------------------------

// Initialize a little-endian TIFF file header.

void header32::init(size_t n)
{
    if (n & 1) n++; // Pad the text length.
    
    endianness = 0x4949;
    version    = 0x002A;
    first_ifd  = uint32_t(sizeof (header32) + n);
}

void header64::init(size_t n)
{
    if (n & 1) n++; // Pad the text length.
    
    endianness = 0x4949;
    version    = 0x002B;
    offsetsize = 0x0008;
    zero       = 0x0000;
    first_ifd  = uint64_t(sizeof (header64) + n);
}

//------------------------------------------------------------------------------

// Initialize a TIFF image file directory field.

void field32::init(int t, int y, size_t c, off_t o)
{
    tag    = uint16_t(t);
    type   = uint16_t(y);
    count  = uint32_t(c);
    offset = uint32_t(o);
}

void field64::init(int t, int y, size_t c, off_t o)
{
    tag    = uint16_t(t);
    type   = uint16_t(y);
    count  = uint64_t(c);
    offset = uint64_t(o);
}

//------------------------------------------------------------------------------

// Initialize a TIFF image file directory. All magic numbers are defined in the
// TIFF Revision 6.0 specification.

// Arguments
//     o : file offset
//     d : data size
//     t : text size
//     s : page size
//     c : channels per pixel
//     b : bits per channel

void ifd32::init(off_t o, size_t d, size_t t, size_t s, size_t c, size_t b)
{
    count = 15;
    next  =  0;

    bits_per_sample.  init(0x0102,  3,  c, o + offsetof (ifd32, bps_value));
    sub_ifds.         init(0x014A,  4,  4, o + offsetof (ifd32, ifd_value));
    datetime.         init(0x0132,  2, 20, o + offsetof (ifd32, date_str));
    strip_offsets.    init(0x0111,  4,  1, o +   sizeof (ifd32));
    subfile_type.     init(0x00FE,  4,  1, 2);
    image_width.      init(0x0100,  3,  1, s);
    image_length.     init(0x0101,  3,  1, s);
    compression.      init(0x0103,  3,  1, 8);
    interpretation.   init(0x0106,  3,  1, c == 1 ? 1 : 2);
    orientation.      init(0x0112,  3,  1, 2);
    samples_per_pixel.init(0x0115,  3,  1, c);
    strip_byte_counts.init(0x0117,  4,  1, d);
    configuration.    init(0x011C,  3,  1, 1);
    predictor.        init(0x013D,  3,  1, 2);
    copyright.        init(0x8298,  2,  t, sizeof (header32));

    bps_value[0] = (0 < c) ? b : 0;
    bps_value[1] = (1 < c) ? b : 0;
    bps_value[2] = (2 < c) ? b : 0;
    bps_value[3] = (3 < c) ? b : 0;

    ifd_value[0] = 0;
    ifd_value[1] = 0;
    ifd_value[2] = 0;
    ifd_value[3] = 0;

    // Encode the current time.

    const time_t now = time(0);
    strftime(date_str, 20, "%Y:%m:%d %H:%M:%S", localtime(&now));

    // Accommodate a bits-per-sample value that fits within an offset.

    if (c * sizeof (uint16_t) <= sizeof (uint32_t))
        memcpy(&bits_per_sample.offset, bps_value, sizeof (uint32_t));
}

void ifd64::init(off_t o, size_t d, size_t t, size_t s, size_t c, size_t b)
{
    count = 15;
    next  =  0;

    bits_per_sample.  init(0x0102,  3,  c, o + offsetof (ifd64, bps_value));
    sub_ifds.         init(0x014A, 18,  4, o + offsetof (ifd64, ifd_value));
    datetime.         init(0x0132,  2, 20, o + offsetof (ifd64, date_str));
    strip_offsets.    init(0x0111, 16,  1, o +   sizeof (ifd64));
    subfile_type.     init(0x00FE,  4,  1, 2);
    image_width.      init(0x0100,  3,  1, s);
    image_length.     init(0x0101,  3,  1, s);
    compression.      init(0x0103,  3,  1, 8);
    interpretation.   init(0x0106,  3,  1, c == 1 ? 1 : 2);
    orientation.      init(0x0112,  3,  1, 2);
    samples_per_pixel.init(0x0115,  3,  1, c);
    strip_byte_counts.init(0x0117,  4,  1, d);
    configuration.    init(0x011C,  3,  1, 1);
    predictor.        init(0x013D,  3,  1, 2);
    copyright.        init(0x8298,  2,  t, sizeof (header64));

    bps_value[0] = (0 < c) ? b : 0;
    bps_value[1] = (1 < c) ? b : 0;
    bps_value[2] = (2 < c) ? b : 0;
    bps_value[3] = (3 < c) ? b : 0;

    ifd_value[0] = 0;
    ifd_value[1] = 0;
    ifd_value[2] = 0;
    ifd_value[3] = 0;

    // Encode the current time.

    const time_t now = time(0);
    strftime(date_str, 20, "%Y:%m:%d %H:%M:%S", localtime(&now));

    // Accommodate a bits-per-sample value that fits within an offset.

    if (c * sizeof (uint16_t) <= sizeof (uint64_t))
        memcpy(&bits_per_sample.offset, bps_value, sizeof (uint64_t));
}

//------------------------------------------------------------------------------

tiff::tiff(const std::string& name,
           const std::string& copyright, size_t s, size_t c, size_t b, bool big)
    : prev(0), text(copyright.size() + 1), big(big)
{
    // Initialize the compression scratch buffer.

    buff = malloc(compressBound(s * s * c * b / 8));

    // Open a new TIFF file and write the header.

    if ((file = fopen(name.c_str(), "w+b")))
    {
        if (big)
        {
            header64 h;
            h.init(text);
            fwrite(&h, sizeof (header64), 1, file);
        }
        else
        {
            header32 h;
            h.init(text);
            fwrite(&h, sizeof (header32), 1, file);
        }
            
        fwrite(copyright.c_str(), copyright.size() + 1, 1, file);
        pad();
    }
}

tiff::~tiff()
{
    next(prev, 0);
    free(buff);
    fclose(file);
}

//------------------------------------------------------------------------------

// Compress the given raster and append it to the file as a new IFD. Return the
// offset of the new IFD to enable SubIFD linkage by the caller. WARNING: the
// contents of the raster will be scrambled by the horizontal difference.

off_t tiff::append(raster *data)
{
    off_t o;
    
    if ((o = ftello(file)) != -1)
    {
        size_t slen = data->length();
        size_t dlen = compressBound(slen);

        const Bytef *sptr = (const Bytef *) data->buffer(0);
              Bytef *dptr =       (Bytef *) buff;

        // Apply the horizontal predector and compress the data raster.
        
        data->hdiff();
        
        if (compress2(dptr, &dlen, sptr, slen, 9) == Z_OK)
        {
            // Initialize and write a TIFF IFD.
        
            if (big)
            {
                ifd64 d;
                d.init(o, dlen, text, data->getw(), data->getc(), data->getb());            
                fwrite(&d, sizeof (ifd64), 1, file);
            }
            else
            {
                ifd32 d;
                d.init(o, dlen, text, data->getw(), data->getc(), data->getb());            
                fwrite(&d, sizeof (ifd32), 1, file);
            }
            
            // Write the compressed data and pad.
            
            fwrite(dptr, dlen, 1, file);
            pad();

            // Link the previous IFD to the new one.

            next(prev, o);
            prev = o;
            fflush(file);
            return o;
        }
    }
    return -1;
}

// Set page c to be a child of page p by modifying the IFD at offset p to
// include the value c as the ith element of the SubIFDs entry. Leave the
// file pointer at the end to ensure the next write appends.

void tiff::link(off_t p, off_t c, int i)
{
    if (p)
    {
        if (big)
        {
            ifd64 d;
            get_ifd(p, d);
            d.ifd_value[i] = uint64_t(c);
            put_ifd(p, d);
        }
        else
        {
            ifd32 d;
            get_ifd(p, d);
            d.ifd_value[i] = uint32_t(c);
            put_ifd(p, d);
        }
    }
    ffd();
}

// Set page p to be the predecessor of page n by modifying the IFD at offset p
// to have a next offset value of n. Leave the file pointer at the end to
// ensure the next write appends.

void tiff::next(off_t p, off_t n)
{
    if (p)
    {
        if (big)
        {
            ifd64 d;
            get_ifd(p, d);
            d.next = uint64_t(n);
            put_ifd(p, d);
        }
        else
        {
            ifd32 d;
            get_ifd(p, d);
            d.next = uint32_t(n);
            put_ifd(p, d);
        }
    }
    ffd();
}

//------------------------------------------------------------------------------

// Read an IFD to a file at the given offset. This will move the file pointer
// to the end of the requested IFD.

template <typename ifd> bool tiff::get_ifd(off_t offset, ifd& d)
{
    if (fseeko(file, offset, SEEK_SET) == 0)

        if (fread(&d, sizeof (ifd), 1, file) == 1)

            return true;

    return false;
}

// Write an IFD to a file at the given offset. This will move the file pointer
// to the end of the written IFD.

template <typename ifd> bool tiff::put_ifd(off_t offset, ifd& d)
{
    if (fseeko(file, offset, SEEK_SET) == 0)

        if (fwrite(&d, sizeof (ifd), 1, file) == 1)

            return true;

    return false;
}

//------------------------------------------------------------------------------

// Move the given file pointer to the end of the file.

void tiff::ffd()
{
    fseeko(file, 0, SEEK_END);
}

// Ensure that the current length of the file is even (16-bit padded).

void tiff::pad()
{
    off_t o;
    
    while ((o = ftello(file)) != -1 && (o % 2))
        fputc(0, file);
}

//------------------------------------------------------------------------------
