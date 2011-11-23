/***************************************************************
* File Name: CAVEGroupIconToolkit.h
*
* Description: Derived class from CAVEGroup, decendent of
* 'DesignObjectHandler::mCAVEGeodeIconToolkitSwitch' as container
* of CAVEGeodeIconToolkit objects.
*
***************************************************************/

#ifndef _CAVE_GROUP_ICON_TOOLKIT_H_
#define _CAVE_GROUP_ICON_TOOLKIT_H_


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
#include "CAVEGeodeIconToolkit.h"
#include "CAVEGroup.h"

typedef std::vector<osg::MatrixTransform*> MatrixTransVector;

/***************************************************************
* Class: CAVEGroupIconToolkit
***************************************************************/
class CAVEGroupIconToolkit: public CAVEGroup
{
  public:

    CAVEGroupIconToolkit(const CAVEGeodeIconToolkit::Type &typ);

    /* set initial bouding radius of the wireframe used in manipulator, only called in 'DOGeometryCollector' */
    static void setManipulatorBoundRadius(const osg::BoundingBox& bb);

    /* set bouding sizes of the wireframe used in manipulator, called in 'DOGeometryCollector' and 'DOGeometryEditor' */
    static void setManipulatorBound(const osg::BoundingBox& bb);

  protected:

    CAVEGeodeIconToolkit::Type mType;

    /* CAVEGeodeIconToolkit vector that contains editting icons within the group */
    CAVEGeodeIconVector mCAVEGeodeIconVector;
    MatrixTransVector mMatrixTransVector;

    /* group init functions that create vectors of 'CAVEGeodeIconToolkit' objects */
    void initGroupMove();
    void initGroupClone();
    void initGroupRotate();
    void initGroupManipulate();

    /* initial bounding radius of manipulator in Design State space: Values set only when the first geode is selected */
    static float gManipulatorBoundRadius;

    /* instance pointers of 'CAVEGroupIconToolkit': each class is restricted to only one tracable instance */
    static CAVEGroupIconToolkit *gMoveInstancePtr;
    static CAVEGroupIconToolkit *gCloneInstancePtr;
    static CAVEGroupIconToolkit *gRotateInstancePtr;
    static CAVEGroupIconToolkit *gManipulateInstancePtr;
};


#endif









