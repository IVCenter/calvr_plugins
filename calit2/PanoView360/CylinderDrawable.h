#ifndef _CYLINDER_DRAWABLE_H_
#define _CYLINDER_DRAWABLE_H_

#include "PanoDrawable.h"

class CylinderDrawable : public PanoDrawable
{
  public:
    CylinderDrawable(float radius_in, float viewanglev_in,float viewangleh_in, float camHeight_in, int segmentsPerTexture_in, int maxTextureSize_in);
    ~CylinderDrawable();

    /** Copy constructor using CopyOp to manage deep vs shallow copy.*/
    CylinderDrawable(const CylinderDrawable&,const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);
    virtual Object* cloneType() const { return new CylinderDrawable(30000, 120, 360.0, 0.0, 100, 4096); }
    virtual Object* clone(const osg::CopyOp& copyop) const { return new CylinderDrawable(*this,copyop); }
    virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const CylinderDrawable*>(obj)!=NULL; }
    virtual const char* libraryName() const { return "Cylinder"; }
    virtual const char* className() const { return "CylinderDrawable"; }

    virtual void drawShape(PanoDrawable::eye eye, int context) const;
  protected:

};
#endif
