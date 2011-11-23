/***************************************************************
* File Name: CAVEGeodeEditWireframe.h
*
* Class Name: CAVEGeodeEditWireframe
*
***************************************************************/

#ifndef _CAVE_GEODE_EDIT_WIREFRAME_H_
#define _CAVE_GEODE_EDIT_WIREFRAME_H_


// C++
#include <iostream>
#include <list>

// Open scene graph
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/StateSet>

// local
#include "CAVEGeometry.h"
#include "CAVEGeode.h"


/***************************************************************
* Class: CAVEGeodeEditWireframe
***************************************************************/
class CAVEGeodeEditWireframe: public CAVEGeode
{
  public:
    CAVEGeodeEditWireframe();
    ~CAVEGeodeEditWireframe();

    virtual void movedon() {}
    virtual void movedoff() {}
    virtual void pressed() {}
    virtual void released() {}

    static const float gSnappingUnitDist;

  protected:

    void initUnitWireBox();

    osg::Geometry *mGeometry;
};


/***************************************************************
* Class: CAVEGeodeEditWireframeMove
***************************************************************/
class CAVEGeodeEditWireframeMove: public CAVEGeodeEditWireframe
{
  public:
    CAVEGeodeEditWireframeMove();
};


/***************************************************************
* Class: CAVEGeodeEditWireframeRotate
***************************************************************/
class CAVEGeodeEditWireframeRotate: public CAVEGeodeEditWireframe
{
  public:
    CAVEGeodeEditWireframeRotate();
};


/***************************************************************
* Class: CAVEGeodeEditWireframeManipulate
***************************************************************/
class CAVEGeodeEditWireframeManipulate: public CAVEGeodeEditWireframe
{
  public:
    CAVEGeodeEditWireframeManipulate();
};


/***************************************************************
* Class: CAVEGeodeEditGeometryWireframe
*
* Wireframe geode that used during geometry level editting
*
***************************************************************/
class CAVEGeodeEditGeometryWireframe: public CAVEGeodeEditWireframe
{
  public:
    CAVEGeodeEditGeometryWireframe(CAVEGeometry *geometry);

    CAVEGeometry *getCAVEGeometryPtr() { return mCAVEGeometry; }

  protected:
    CAVEGeometry *mCAVEGeometry;	// geometry that used to render wireframe
};


#endif
















