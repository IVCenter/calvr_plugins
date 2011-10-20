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

#ifndef PANOPROC_SAMPLER_STOCHASTIC_HPP
#define PANOPROC_SAMPLER_STOCHASTIC_HPP

#include "sampler.hpp"
#include "poisson.hpp"
#include "image.hpp"

//------------------------------------------------------------------------------

class sampler_stochastic : public sampler
{
public:

    sampler_stochastic() : point(10000, 0.15f, 0.999f) { }

    virtual void get(double *, int, int, const double *, const double *,
                                         const double *, const double *);

private:

    void get(double *, int, int, int, const double *, const double *,
                                      const double *, const double *);

    poisson point;
};

//------------------------------------------------------------------------------

#endif
