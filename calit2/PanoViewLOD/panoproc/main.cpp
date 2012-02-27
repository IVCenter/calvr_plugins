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

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "image_rect.hpp"
#include "image_pds.hpp"

#include "sampler_single.hpp"
#include "sampler_quincunx.hpp"
#include "sampler_adaptive.hpp"
#include "sampler_stochastic.hpp"
#include "sampler_testpattern.hpp"

#include "error.hpp"
#include "panoproc.hpp"

//------------------------------------------------------------------------------

sampler *samp(const char *arg)
{
    if (!strcmp(arg, "single"))     return new sampler_single    ();
    if (!strcmp(arg, "quincunx"))   return new sampler_quincunx   ();
    if (!strcmp(arg, "adaptive"))   return new sampler_adaptive   ();
    if (!strcmp(arg, "stochastic")) return new sampler_stochastic ();
    if (!strcmp(arg, "test"))       return new sampler_testpattern();
    return 0;
}

image_data *imag(const char *arg)
{
    if (!strncmp(arg, "rect", 4)) return new image_rect(arg + 4);
    if (!strncmp(arg, "pds",  3)) return new image_pds();
    return 0;
}

int main(int argc, char *argv[])
{
    set_executable(argv[0]);
    
    if (argc > 1)
    {
        const char *t = "Copyright (c) 2011 Robert Kooima";
        const char *o = "out.tif";
        int         n = 512;
        int         d = 0;
        bool        b = false;
        bool        c = false;
    
        image_data *I = 0;
        sampler    *S = 0;
        panoproc   *P = 0;
        
        // Parse all options arguments.
        
        int argi = 1;
        
        for (; argi < argc; ++argi)
            if      (!strcmp(argv[argi], "-o")) o =      argv[++argi];
            else if (!strcmp(argv[argi], "-t")) t =      argv[++argi];
            else if (!strcmp(argv[argi], "-n")) n = atoi(argv[++argi]);
            else if (!strcmp(argv[argi], "-d")) d = atoi(argv[++argi]);
            else if (!strcmp(argv[argi], "-S")) S = samp(argv[++argi]);
            else if (!strcmp(argv[argi], "-I")) I = imag(argv[++argi]);
            else if (!strcmp(argv[argi], "-b")) b = true;
            else if (!strcmp(argv[argi], "-c")) c = true;
            else break;

        // Instance unconfigured defaults.

        if (S == 0) S = samp("single");
        if (I == 0) I = imag("rect");

        // All non-option arguments name input files.

        for (; argi < argc; ++argi)
            I->source(argv[argi]);

        // Do it.

        P = new panoproc(S);

        S->setimage(I);
        I->setcubic(c);
        P->write(o, t, n, d, b);
        
        delete P;
        delete S;
        delete I;
    }
    else app_log_mesg("\nUsage: %s [opt] infile ...\n"
                      "\t-o <name> ... Output file name    (out.tif)\n"
                      "\t-n <n>    ... Output page size    (512)\n"
                      "\t-d <d>    ... Subdivision depth   (0)\n"
                      "\t-S <mode> ... Sample mode         (single)\n"
                      "\t-I <type> ... Image type          (rect)\n"
                      "\t-t <text> ... Copyright text\n"
                      "\t-b        ... BigTIFF output\n"
                      "\t-c        ... Bicubic sampling\n"
                      "Sample modes: "
                      "single, "
                      "quincunx, "
                      "adaptive, "
                      "stochastic, "
                      "test\n", argv[0]);
    return 0;
}

//------------------------------------------------------------------------------
