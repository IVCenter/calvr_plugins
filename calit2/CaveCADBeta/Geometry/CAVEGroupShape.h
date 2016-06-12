/***************************************************************
* File Name: CAVEGroupShape.h
*
* Description: Derived class from CAVEGroup, direct descendant
* of 'DesignObjectHandler::mCAVEGeodeShapeSwitch' as container
* of CAVEGeodeShape objects.
*
***************************************************************/

#ifndef _CAVE_GROUP_SHAPE_H_
#define _CAVE_GROUP_SHAPE_H_


// C++
#include <iostream>
#include <list>
#include <string>

// Open scene graph
#include <osg/Geode>
#include <osg/Group>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Switch>
#include <osg/Texture2D>
#include <osg/Vec3>

#include <osgDB/ReadFile>
#include <osgText/Text3D>

// local
#include "CAVEGeodeShape.h"
#include "CAVEGroup.h"


/***************************************************************
* Class: CAVEGroupShape
***************************************************************/
class CAVEGroupShape: public CAVEGroup
{
  public:
    CAVEGroupShape();
    CAVEGroupShape(CAVEGeodeShape *shape);

    /* functions prototypes of attaching & detaching multiple CAVEGeode shapes */
    void addCAVEGeodeShape(CAVEGeodeShape *shape);

    const int getNumCAVEGeodeShapes() { return mCAVEGeodeShapeVector.size(); }
    CAVEGeodeShape *getCAVEGeodeShape(const int &idx);

  protected:
    CAVEGeodeShapeVector mCAVEGeodeShapeVector;
};


#endif

