/***************************************************************
* File Name: ANIMVirtualSphere.h
*
* Description:
*
***************************************************************/

#ifndef _ANIM_VIEWPOINTS_H_
#define _ANIM_VIEWPOINTS_H_


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
#include <osgText/Text>

// Local includes
#include "AnimationModelerBase.h"
#include <cvrKernel/CalVR.h>


namespace CAVEAnimationModeler
{
    void ANIMCreateViewpoints(std::vector<osg::PositionAttitudeTransform*>* fwdVec, 
                              std::vector<osg::PositionAttitudeTransform*>* bwdVec);
    void ANIMAddViewpoint(std::vector<osg::PositionAttitudeTransform*>* fwdVec, 
                          std::vector<osg::PositionAttitudeTransform*>* bwdVec);

};


#endif
