/***************************************************************
* File Name: ANIMGeometryCollector.h
*
* Description:
*
***************************************************************/

#ifndef _ANIM_GEOMETRY_COLLECTOR_H_
#define _ANIM_GEOMETRY_COLLECTOR_H_


// C++
#include <iostream>
#include <list>

// Open scene graph
#include <osg/BlendFunc>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Point>
#include <osg/PointSprite>
#include <osg/PositionAttitudeTransform>
#include <osg/Shader>
#include <osg/ShapeDrawable>
#include <osg/StateAttribute>
#include <osg/StateSet>
#include <osg/Switch>
#include <osg/Texture2D>
#include <osg/Vec3>
#include <osgDB/ReadFile>

// Local includes
#include "AnimationModelerBase.h"


namespace CAVEAnimationModeler
{

    #define ANIM_GEOMETRY_COLLECTOR_SURFACE_PICKUP_TIME		0.2f
    #define ANIM_GEOMETRY_COLLECTOR_SURFACE_PICKUP_SAMPS	16


    /***************************************************************
    * Function: ANIMLoadGeometryCollectorIconSurfaces()
    ***************************************************************/
    void ANIMLoadGeometryCollectorIconSurfaces(osg::PositionAttitudeTransform** surfacesPATrans, 
		CAVEGroupIconSurface **iconSurfaceGroup, CAVEGeodeShape *shapeGeode, CAVEGeodeShape *shapeGeodeRef);

    /***************************************************************
    * Function: ANIMLoadGeometryCollectorGeodeWireframe()
    ***************************************************************/
    void ANIMLoadGeometryCollectorGeodeWireframe(osg::MatrixTransform **wireframeTrans, 
		CAVEGroupEditGeodeWireframe **editGeodeWireframe, CAVEGeodeShape *shapeGeode);

    /***************************************************************
    * Function: ANIMLoadGeometryCollectorGeometryWireframe()
    ***************************************************************/
    void ANIMLoadGeometryCollectorGeometryWireframe(osg::MatrixTransform **wireframeTrans, 
		CAVEGroupEditGeometryWireframe **editGeometryWireframe, CAVEGeometry *geometry);
};


#endif




