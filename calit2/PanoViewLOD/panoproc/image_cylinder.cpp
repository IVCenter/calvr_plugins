#include "image_cylinder.hpp"

#include <cmath>
#include <cstdio>

image_cylinder::image_cylinder(const char * arg) : image_basic(""), VFOV(170.0), HFOV(360.0)
{
    float vert,hor;
    if (sscanf(arg, "%fx%f", &vert, &hor) == 2)
    {
        VFOV = vert;
        HFOV = hor;
    }

    VFOV *= M_PI / 180.0;
    HFOV *= M_PI / 180.0;

    VRange = cos((M_PI - VFOV)/2.0);
    HRange = (1.0 - ((2.0 * M_PI - HFOV) / (2.0 * M_PI))) * M_PI;
}

void image_cylinder::get(double *p, const double *v)
{
    bool inRange = false;

    double i,j;

    if(fabs(v[1]) <= VRange)
    {
	i = (-v[1] / (2.0 * VRange) + 0.5) * (h-1);

	double lon = atan2(v[0], -v[2]);
	if(fabs(lon) <= HRange)
	{
	    j = 0.5 * (HRange + lon) / HRange * (w-1);
	    inRange = true;
	}
    } 

    if(inRange)
    {
	filter(0, i, j, p);
    }
    else
    {
	switch (getc())
	{
	    case 3: p[2] = 0.0;
	    case 2: p[1] = 0.0;
	    case 1: p[0] = 0.0;
	}
    }
}
