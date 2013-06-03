/***************************************************************
* File Name: CAVEGeodeReference.h
*
* Description: Derived class from CAVEGeode
*
***************************************************************/

#ifndef _CAVE_GEODE_REFERENCE_H_
#define _CAVE_GEODE_REFERENCE_H_


// C++
#include <iostream>
#include <list>

// Open scene graph
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Quat>
#include <osg/ShapeDrawable>
#include <osg/StateSet>
#include <osg/Texture2D>
#include <osg/Vec3>

#include <osgDB/ReadFile>

// local
#include "CAVEGeode.h"
#include "CAVEGeodeSnapWireframe.h"
#include "../SnapLevelController.h"


/***************************************************************
* Class: CAVEGeodeReference
***************************************************************/
class CAVEGeodeReference: public CAVEGeode
{
  public:
    CAVEGeodeReference() {}

    virtual void movedon() {}
    virtual void movedoff() {}
    virtual void pressed() {}
    virtual void released() {}
};


/***************************************************************
* Class: CAVEGeodeReferenceAxis
***************************************************************/
class CAVEGeodeReferenceAxis: public CAVEGeodeReference
{
  public:
    CAVEGeodeReferenceAxis();

    /* types of axis orientation */
    enum AxisType
    {
	POS_X,
	POS_Y,
	POS_Z,
	NEG_X,
	NEG_Y,
	NEG_Z
    };

    void setType(const AxisType &typ, osg::MatrixTransform **matTrans);
    void resize(const float &length);

    static const int gHeadUnitSegments;

  protected:
    static const float gBodyRadius;
    static const float gArrowRadius;
    static const float gArrowLength;

    AxisType mType;
    osg::Cone *mCone;
    osg::Cylinder *mCylinder;
    osg::ShapeDrawable *mConeDrawable, *mCylinderDrawable;
};


/***************************************************************
* Class: CAVEGeodeReferencePlane
***************************************************************/
class CAVEGeodeReferencePlane: public CAVEGeodeReference
{
  public:
    CAVEGeodeReferencePlane();

    enum GridColor
    {
	RED,
	GREEN,
	BLUE
    };

    void resize(const float lx, const float ly, const float unitsize);
    void setGridColor(const GridColor &color);
    void setAlpha(const float &alpha);

    static const float gSideLength;

  protected:
    osg::Vec3Array* mVertexArray;
    osg::Vec3Array* mNormalArray;
    osg::Vec2Array* mTexcoordArray;
    osg::Geometry *mGeometry;
};



#endif

