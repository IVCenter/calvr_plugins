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
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <regex.h>

#include "pds.hpp"

//------------------------------------------------------------------------------

static inline std::string match(const std::string& str,
                                const std::string& key,
                                const std::string& val)
{
    static char       pat[256];
    static regmatch_t match[2];
    static regex_t    regex;

    sprintf(pat, "%s *= *%s", key.c_str(), val.c_str());
    
    if (regcomp(&regex, pat, REG_EXTENDED) == 0)
    {
        if (regexec(&regex, str.c_str(), 2, match, 0) == 0)
            return std::string(str, match[1].rm_so,
                   match[1].rm_eo - match[1].rm_so);
    }

    return "";
}

//------------------------------------------------------------------------------

std::string pds_string(const std::string& str, const std::string& key)
{
    std::string val;
    
    // A string surrounded by double quotes may contain whitespace.

    if (!(val = match(str, key, "\"([^\"]+)\"")).empty())
        return val;

    // A string with no double quotes may not contain whitespace.

    if (!(val = match(str, key, "([^\n ]+)")).empty())
        return val;

    return "";
}

int pds_length(const std::string& str, const std::string& key)
{
    // An integer length is usually a buffer size or record count.
    
    std::string val = match(str, key, "([0123456789]+)");
    
    if (!val.empty())
        return int(strtol(val.c_str(), 0, 0));
    else
        return 0;
}

double pds_angle(const std::string& str, const std::string& key)
{
    // An angle is a value followed by a degree or radian token.
    
    std::string val = match(str, key, "(-?[\\.0123456789]+) <[^>]+>");
    std::string dim = match(str, key,  "-?[\\.0123456789]+ <([^>]+)>");
    
    if (!val.empty())
    {
        if (dim == "deg")
            return strtod(val.c_str(), 0) * M_PI / 180.0;
        else
            return strtod(val.c_str(), 0);
    }
    else
        return 0;
}

double pds_value(const std::string& str, const std::string& key)
{
    // Any random jumble of digits and decimals is accepted as a value.
    
    std::string val = match(str, key, "(-?[\\.0123456789]+)");

    if (!val.empty())
        return strtod(val.c_str(), 0);
    else
        return 0;
}

//------------------------------------------------------------------------------
