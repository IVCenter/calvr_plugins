/***************************************************************
* File Name: ANIMVirtualEarth.h
*
* Description:
*
***************************************************************/

#ifndef _ANIM_VIRTUAL_EARTH_H_
#define _ANIM_VIRTUAL_EARTH_H_


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
    #define ANIM_VIRTUAL_SEASONS_MAP_RADIUS	0.25
    #define ANIM_VIRTUAL_SEASONS_MAP_ALPHA	0.8

    void ANIMLoadVirtualEarthReferenceLevel(osg::Switch **designStateSwitch, osg::Geode **seasonsMapNode);
    void ANIMLoadVirtualEarthEclipticLevel(osg::Switch **eclipticSwitch);
    void ANIMLoadVirtualEarthEquatorLevel(osg::Switch **equatorSwitch, osg::Geode **earthGeode, 
		osg::PositionAttitudeTransform **xformTransFwd, osg::PositionAttitudeTransform **xformTransBwd, 
		osg::MatrixTransform **fixedPinIndicatorTrans, osg::MatrixTransform **fixedPinTrans);
    const osg::Vec3 ANIMVirtualEarthLightDir();
    osg::Geode* ANIMCreateWiredSphereGeode(const int numLatiSegs, const int numLongiSegs, 
					   const float rad, const osg::Vec4 color);
};


#endif
