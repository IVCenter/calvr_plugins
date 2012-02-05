/***************************************************************
* File Name: CAVEGroupIconSurface.h
*
* Description: Derived class from CAVEGroup, decendent of
* 'DesignObjectHandler::mCAVEGeodeIconSurfaceSwitch' as container
* of CAVEGeodeIconSurface objects.
*
***************************************************************/

#ifndef _CAVE_GROUP_ICON_SURFACE_H_
#define _CAVE_GROUP_ICON_SURFACE_H_


// C++
#include <iostream>
#include <list>
#include <string>
#include <vector>

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
#include "CAVEGeodeIconSurface.h"
#include "CAVEGeodeShape.h"
#include "CAVEGroup.h"
#include "CAVEGroupEditWireframe.h"


class CAVEGroupIconSurface;
typedef std::vector<CAVEGroupIconSurface*>	CAVEGroupIconSurfaceVector;


/***************************************************************
* Class: CAVEGroupIconSurface
***************************************************************/
class CAVEGroupIconSurface: public CAVEGroup
{
  public:
    CAVEGroupIconSurface();

    /* Function 'acceptCAVEGeodeShape()': parses geometry info contained in one
       'CAVEGeodeShape' object and generate a vector of 'CAVEGeodeIconSurface',
       it also correlates reference geometry pointers to those in 'shapeGeodeRef'.
    */
    void acceptCAVEGeodeShape(CAVEGeodeShape *shapeGeode, CAVEGeodeShape *shapeGeodeRef);

    /* icon surface group highlighting functions: access to function calls with the same name in 'CAVEGeodeIconSurface' */
    void setHighlightNormal();
    void setHighlightSelected();
    void setHighlightUnselected();

    /* final morphing functions called by 'DOGeometryEditor' */
    void applyEditingFinishes(CAVEGeodeShape::EditorInfo **infoPtr);

    /* set/get the two centers for translation: 'CAVEGeodeIcon' is translated from 'gShapeCenter' to 'gIconCenter' */
    static void setSurfaceTranslationHint(const osg::Vec3 &shapeCenter, const osg::Vec3 &iconCenter);
    static void getSurfaceTranslationHint(osg::Vec3 &shapeCenter, osg::Vec3 &iconCenter);
    static void getSurfaceTranslationIconCenterHint(osg::Vec3 &iconCenter);

    /* set/get the scale value for generating icon surface object based on the size of first hit 'CAVEGeodeShape' */
    static void setSurfaceScalingHint(const osg::BoundingBox& boundingbox);
    static const float &getSurfaceScalingHint();

  protected:
    /* each surface geometry is decomposed into a geode in this vector */
    CAVEGeodeIconVector mCAVEGeodeIconVector;

    /* 'CAVEGeodeShape' pointer used to track the origin 'CAVEGeodeShape' that this group refers to */
    CAVEGeodeShape *mCAVEGeodeShapeOriginPtr;

    /* matrix transform that applies offset from 'gShapeCenter' to Vec3(0, 0, 0) */
    osg::MatrixTransform *mRootTrans;

    /* vertex, normal and texcoord data array that shared by all 'CAVEGeodeIcon' objects, which are exactly
       the same as those in actual CAVEGeodeShape data arrays */
    osg::Vec3Array *mSurfVertexArray;
    osg::Vec3Array *mSurfNormalArray;
    osg::Vec3Array *mSurfUDirArray,  *mSurfVDirArray;
    osg::Vec2Array *mSurfTexcoordArray;

    bool mPrimaryFlag;

    /* static data members that used to resize and replace the icon surface objects */
    static osg::Vec3 gShapeCenter;
    static osg::Vec3 gIconCenter;
    static float gScaleVal;
};


#endif



