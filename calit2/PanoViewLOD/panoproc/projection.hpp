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

#ifndef PANOPROC_PROJECTION_HPP
#define PANOPROC_PROJECTION_HPP

#include <cmath>
#include <cstdio>
#include "pds.hpp"

//------------------------------------------------------------------------------

#define DEBUGPRINT printf

class projection
{
public:

    projection(const std::string& in) :

        latmax(pds_angle(in.c_str(),     "MAXIMUM_LATITUDE")),
        latmin(pds_angle(in.c_str(),     "MINIMUM_LATITUDE")),
        latp  (pds_angle(in.c_str(),      "CENTER_LATITUDE")),
        lonmax(pds_angle(in.c_str(), "EASTERNMOST_LONGITUDE")),
        lonmin(pds_angle(in.c_str(), "WESTERNMOST_LONGITUDE")),
        lonp  (pds_angle(in.c_str(),      "CENTER_LONGITUDE")),

        l0    (pds_value(in.c_str(),   "LINE_PROJECTION_OFFSET")),
        s0    (pds_value(in.c_str(), "SAMPLE_PROJECTION_OFFSET")),

        res   (pds_value(in.c_str(), "MAP_RESOLUTION")),
        scale (pds_value(in.c_str(), "MAP_SCALE")),
        radius(pds_value(in.c_str(), "A_AXIS_RADIUS") * 1000.0)
    {
        DEBUGPRINT("latmax %f\n", latmax);
        DEBUGPRINT("latmin %f\n", latmin);
        DEBUGPRINT("latp   %f\n", latp);
        DEBUGPRINT("lonmax %f\n", lonmax);
        DEBUGPRINT("lonmin %f\n", lonmin);
        DEBUGPRINT("lonp   %f\n", lonp);
        DEBUGPRINT("l0     %f\n", l0);
        DEBUGPRINT("s0     %f\n", s0);
        DEBUGPRINT("res    %f\n", res);
        DEBUGPRINT("scale  %f\n", scale);
        DEBUGPRINT("radius %f\n", radius);
    }
    
    virtual void torowcol(double, double, double&, double&) const = 0;
    virtual void tolatlon(double, double, double&, double&) const = 0;

    bool iswithin(double lat, double lon) const
    {
        if (latmin <= lat && lat < latmax)
        {
            if (lonmin < lonmax)
                return (lonmin <= lon && lon < lonmax);
            else
                return (lonmin <  lon || lon <= lonmax);
        }
        return false;
    }


protected:

    double tolon(double a) const
    {
        double b = fmod(a, 2 * M_PI);
        return b < 0 ? b + 2 * M_PI : b;
    }

    double todeg(double r) const
    {
        return r * 180.0 / M_PI;
    }

    double torad(double d) const
    {
        return M_PI * d / 180.0;
    }

    void toLS(double x, double y, double &l, double &s) const
    {
        l = l0 - y / scale;
        s = s0 + x / scale;
    }
    
    void toXY(double l, double s, double&x, double& y) const
    {
        x = (s - s0) * scale;
        y = (l0 - l) * scale;
    }
    
    double latmax;
    double latmin;
    double latp;
    double lonmax;
    double lonmin;
    double lonp;
    double l0;
    double s0;
    double res;
    double scale;
    double radius;
};

//------------------------------------------------------------------------------

class equirectangular : public projection
{
public:

    equirectangular(const std::string& str) : projection(str) { }

    virtual void torowcol(double lat, double lon, double& l, double& s) const
    {
        double x = radius * (lon - lonp) * cos(latp);
        double y = radius * (lat);

        toLS(x, y, l, s);
    }

    virtual void tolatlon(double l, double s, double& lat, double& lon) const
    {
        double x;
        double y;
        
        toXY(l, s, x, y);
        
        lat =        y / (radius);
        lon = lonp + x / (radius * cos(latp));
    }
};

//------------------------------------------------------------------------------

class polar_stereographic : public projection
{
public:

    polar_stereographic(const std::string& str) : projection(str) { }

    virtual void torowcol(double lat, double lon, double& l, double& s) const
    {
        double x;
        double y;
        
        if (latp > 0)
        {
            x =  2 * radius * tan(M_PI_4 - lat / 2) * sin(lon - lonp);
            y = -2 * radius * tan(M_PI_4 - lat / 2) * cos(lon - lonp);
        }
        else
        {
            x =  2 * radius * tan(M_PI_4 + lat / 2) * sin(lon - lonp);
            y =  2 * radius * tan(M_PI_4 + lat / 2) * cos(lon - lonp);
        }
        
        toLS(x, y, l, s);
    }

    virtual void tolatlon(double l, double s, double& lat, double& lon) const
    {
        double x;
        double y;
        
        toXY(l, s, x, y);

        double p = sqrt(x * x + y * y);
        double c = atan(radius * p / 2) * 2;

        lat = asin(cos(c) * sin(latp) + y * sin(c) * cos(latp) / p);

        if (latp > 0)
            lon = tolon(lonp + atan2(x, -y));
        else
            lon = tolon(lonp + atan2(x,  y));
    }
};

//------------------------------------------------------------------------------

class orthographic : public projection
{
public:

    orthographic(const std::string& str) : projection(str) { }

    virtual void torowcol(double lat, double lon, double& l, double& s) const
    {
        double x = radius * cos(lat) * sin(lon - lonp);
        double y = radius * sin(lat);
        
        toLS(x, y, l, s);
    }

    virtual void tolatlon(double l, double s, double& lat, double& lon) const
    {
        double x;
        double y;
        
        toXY(l, s, x, y);

        double p = sqrt(x * x + y * y);
        double c = asin(p / radius);

        lat = asin(cos(c) * sin(latp) + y * sin(c) * cos(latp) / p);
        lon = tolon(lonp + atan2(x * sin(c), p * cos(c)));
    }
};

//------------------------------------------------------------------------------

class simple_cylindrical : public projection
{
public:

    simple_cylindrical(const std::string& str) : projection(str) { }

    virtual void torowcol(double lat, double lon, double& l, double& s) const
    {
        s = s0 + res * (todeg(lon) - todeg(lonp));
        l = l0 - res * (todeg(lat) - todeg(latp));
    }

    virtual void tolatlon(double l, double s, double& lat, double& lon) const
    {
        lon = torad(todeg(lonp) + (s - s0) / res);
        lat = torad(todeg(latp) - (l - l0) / res);
    }
};

//------------------------------------------------------------------------------

#endif