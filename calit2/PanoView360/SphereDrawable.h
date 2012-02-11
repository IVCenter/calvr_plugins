#ifndef _SPHERE_DRAWABLE_H_
#define _SPHERE_DRAWABLE_H_

#include <osg/Geometry>
#include <osg/Vec3>
#include <OpenThreads/Mutex>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <glu.h>
#else
#include <GL/glu.h>
#endif
#include <iostream>
#include <vector>

#include "PanoDrawable.h"

#include <osgDB/ReadFile>

class SphereDrawable : public PanoDrawable
{
  public:
    SphereDrawable(float radius_in, float viewanglev_in, float viewangleh_in, float camHeight_in, int segmentsPerTexture_in, int maxTextureSize_in);
    virtual ~SphereDrawable();

    /** Copy constructor using CopyOp to manage deep vs shallow copy.*/
    SphereDrawable(const SphereDrawable&,const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);
    virtual Object* cloneType() const { return new SphereDrawable(30000, 120, 360, 0.0, 100, 4096); }
    virtual Object* clone(const osg::CopyOp& copyop) const { return new SphereDrawable(*this,copyop); }
    virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const SphereDrawable*>(obj)!=NULL; }
    virtual const char* libraryName() const { return "Sphere"; }
    virtual const char* className() const { return "SphereDrawable"; }

    virtual void drawShape(PanoDrawable::eye eye, int context) const;

  protected:

};
#endif
