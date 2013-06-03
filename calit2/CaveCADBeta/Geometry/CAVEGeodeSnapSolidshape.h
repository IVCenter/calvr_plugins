/***************************************************************
* File Name: CAVEGeodeSnapSolidshape.h
*
* Class Name: CAVEGeodeSnapSolidshape
*
***************************************************************/

#ifndef _CAVE_GEODE_SNAP_SOLIDSHAPE_H_
#define _CAVE_GEODE_SNAP_SOLIDSHAPE_H_


// C++
#include <iostream>
#include <list>

// Open scene graph
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/StateSet>

// local
#include "CAVEGeode.h"


/***************************************************************
* Class: CAVEGeodeSnapSolidshape
***************************************************************/
class CAVEGeodeSnapSolidshape: public CAVEGeode
{
  public:
    CAVEGeodeSnapSolidshape();

    virtual void movedon() {}
    virtual void movedoff() {}
    virtual void pressed() {}
    virtual void released() {}

    virtual void resize(const osg::Vec3 &gridVect, bool snap = true) = 0;
    bool isValid();

    /* shape morphing functions: 'dirtyBound()' is called with in 'resize()' */
    void setSnappingUnitDist(const float &dist) { mSnappingUnitDist = dist; }
    void setInitPosition(const osg::Vec3 &initPos, bool snap = true);
    void setScaleVect(const osg::Vec3 &scaleVect) { mScaleVect = scaleVect; }

    const float &getSnappingUnitDist() { return mSnappingUnitDist; }
    const osg::Vec3 &getInitPosition() { return mInitPosition; }
    const osg::Vec3 &getScaleVect() { return mScaleVect; }

  protected:

    /* 'mSnappingUnitDist' in CAVEGeodeSnapSolidshape represents the actual lengths in shape that
	being created. This value is usually different from 'gSnappingUnitDist' of CAVEGeodeSnapWireframe, 
	and varies as SnappingLevelController is switching between different levels. */
    float mSnappingUnitDist;
    osg::Vec3 mInitPosition, mScaleVect;
};


/***************************************************************
* Class: CAVEGeodeSnapSolidshapeBox
***************************************************************/
class CAVEGeodeSnapSolidshapeBox: public CAVEGeodeSnapSolidshape
{
  public:
    CAVEGeodeSnapSolidshapeBox();

    virtual void resize(const osg::Vec3 &gridVect, bool snap);

  protected:
    osg::Box* mBox;
};


/***************************************************************
* Class: CAVEGeodeSnapSolidshapeCylinder
***************************************************************/
class CAVEGeodeSnapSolidshapeCylinder: public CAVEGeodeSnapSolidshape
{
  public:
    CAVEGeodeSnapSolidshapeCylinder();

    virtual void resize(const osg::Vec3 &gridVect, bool snap);

  protected:
    osg::Cylinder* mCylinder;
};


/***************************************************************
* Class: CAVEGeodeSnapSolidshapeCone
***************************************************************/
class CAVEGeodeSnapSolidshapeCone: public CAVEGeodeSnapSolidshape
{
  public:
    CAVEGeodeSnapSolidshapeCone();

    virtual void resize(const osg::Vec3 &gridVect, bool snap);

  protected:
    osg::Cone* mCone;
};


/***************************************************************
* Class: CAVEGeodeSnapSolidshapeLine
***************************************************************/
class CAVEGeodeSnapSolidshapeLine: public CAVEGeodeSnapSolidshape
{
  public:
    CAVEGeodeSnapSolidshapeLine();

    virtual void resize(const osg::Vec3 &gridVect, bool snap);

  protected:
    osg::Box* mBox;
};


#endif

