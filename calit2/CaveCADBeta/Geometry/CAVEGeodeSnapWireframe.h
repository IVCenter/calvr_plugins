/***************************************************************
* File Name: CAVEGeodeSnapWireframe.h
*
* Class Name: CAVEGeodeSnapWireframe
*
***************************************************************/

#ifndef _CAVE_GEODE_SNAP_WIREFRAME_H_
#define _CAVE_GEODE_SNAP_WIREFRAME_H_


// C++
#include <iostream>
#include <list>

// Open scene graph
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/StateSet>

// local
#include "CAVEGeode.h"


/***************************************************************
* Class: CAVEGeodeSnapWireframe
***************************************************************/
class CAVEGeodeSnapWireframe: public CAVEGeode
{
  public:
    CAVEGeodeSnapWireframe();
    ~CAVEGeodeSnapWireframe();

    virtual void movedon() {}
    virtual void movedoff() {}
    virtual void pressed() {}
    virtual void released() {}

    virtual void resize(osg::Vec3 &gridVect) = 0;

    // shape morphing functions: 'dirtyBound()' is called with in 'resize()'
    void setInitPosition(const osg::Vec3 &initVect) { mInitPosition = initVect; }
    void setScaleVect(const osg::Vec3 &scaleVect) { mScaleVect = scaleVect; }

    const osg::Vec3 &getInitPosition() { return mInitPosition; }
    const osg::Vec3 &getScaleVect() { return mScaleVect; }
    const osg::Vec3 &getDiagonalVect() { return mDiagonalVect; }

    static const float gSnappingUnitDist;

  protected:
    virtual void initBaseGeometry() = 0;

    float mSnappingUnitDist;

    osg::Vec3 mInitPosition, mScaleVect, mDiagonalVect;
    osg::Geometry *mBaseGeometry, *mSnapwireGeometry;
};


/***************************************************************
* Class: CAVEGeodeSnapWireframeBox
***************************************************************/
class CAVEGeodeSnapWireframeBox: public CAVEGeodeSnapWireframe
{
  public:
    CAVEGeodeSnapWireframeBox();

    virtual void resize(osg::Vec3 &gridVect);
    virtual void initBaseGeometry();
};


/***************************************************************
* Class: CAVEGeodeSnapWireframeCylinder
***************************************************************/
class CAVEGeodeSnapWireframeCylinder: public CAVEGeodeSnapWireframe
{
  public:
    CAVEGeodeSnapWireframeCylinder();

    virtual void resize(osg::Vec3 &gridVect);
    virtual void initBaseGeometry();

    static const int gMinFanSegments;
    static int gCurFanSegments;
};


#endif

