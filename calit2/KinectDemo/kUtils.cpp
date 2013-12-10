#include <kUtils.h>

// convert depth to hue angle in degrees
double depth_to_hue(double dmin, double depth, double dmax)
{
    double denom = dmax - dmin;
    double numer = depth - dmin;

    if (numer < 0)
        numer = 0;
    else if (numer > denom)
        numer = denom;

    double frac = numer / denom;
    const double offset = 0;    // in degrees
    return fmod(frac * 360 + offset, 360);
}

// http://www.cs.rit.edu/~ncs/color/t_convert.html
void HSVtoRGB(float* r, float* g, float* b, float h, float s, float v)
{
    int i;
    float f, p, q, t;

    if (s == 0) {
        // achromatic (grey)
        *r = *g = *b = v;
        return;
    }

    h /= 60;            // sector 0 to 5
    i = floor(h);
    f = h - i;          // factorial part of h
    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));

    switch (i) {
    case 0:
        *r = v;
        *g = t;
        *b = p;
        break;

    case 1:
        *r = q;
        *g = v;
        *b = p;
        break;

    case 2:
        *r = p;
        *g = v;
        *b = t;
        break;

    case 3:
        *r = p;
        *g = q;
        *b = v;
        break;

    case 4:
        *r = t;
        *g = p;
        *b = v;
        break;

    default:        // case 5:
        *r = v;
        *g = p;
        *b = q;
        break;
    }
}


