#ifndef PANOPROC_IMAGE_CYLINDER_HPP
#define PANOPROC_IMAGE_CYLINDER_HPP

#include "image_basic.hpp"

class image_cylinder : public image_basic
{
public:
    image_cylinder(const char * arg);
    
    virtual void get(double *, const double *);

protected:
    double VFOV;
    double HFOV;
    double VRange;
    double HRange;
};

#endif
