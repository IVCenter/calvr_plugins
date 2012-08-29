/***************************************************************
* File Name: AnimationModeler.h
*
* Description: Interface to a set of animations/models
*
***************************************************************/

#ifndef _ANIMATION_MODELER_H_
#define _ANIMATION_MODELER_H_


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

#include <osgParticle/Particle>
#include <osgParticle/ParticleSystem>
#include <osgParticle/ParticleSystemUpdater>
#include <osgParticle/ModularEmitter>
#include <osgParticle/ModularProgram>
#include <osgParticle/RandomRateCounter>
#include <osgParticle/SectorPlacer>
#include <osgParticle/RadialShooter>
#include <osgParticle/AccelOperator>
#include <osgParticle/FluidFrictionOperator>


// Local includes
#include "../Geometry/CAVEGeodeIcon.h"
#include "../Geometry/CAVEGeodeIconSurface.h"
#include "../Geometry/CAVEGeodeIconToolkit.h"
#include "../Geometry/CAVEGeodeReference.h"
#include "../Geometry/CAVEGeodeSnapWireframe.h"
#include "../Geometry/CAVEGeodeSnapSolidshape.h"
#include "../Geometry/CAVEGeodeShape.h"
#include "../Geometry/CAVEGroupReference.h"
#include "../Geometry/CAVEGroupShape.h"
#include "../Geometry/CAVEGroupIconSurface.h"
#include "../Geometry/CAVEGroupIconToolkit.h"

#include <cvrConfig/ConfigManager.h>


namespace CAVEAnimationModeler
{
    #define ANIM_VIRTUAL_SPHERE_RADIUS		0.2
    #define ANIM_VIRTUAL_SPHERE_DISTANCE	1.0
    #define ANIM_VIRTUAL_SPHERE_LAPSE_TIME	0.5
    #define ANIM_VIRTUAL_SPHERE_NUM_SAMPS	30

    #define ANIM_SKYDOME_RADIUS			400.0f

    #define ANIM_WATERSURF_SIZE			420.0f
    #define ANIM_WATERSURF_ALTITUDE		-2.0f
    #define ANIM_WATERSURF_TEXCOORD_NOISE_LEVEL	0.002f

    #define ANIM_PARA_PAINT_FRAME_ZOOM_FACTOR	1.8f
    #define ANIM_PARA_PAINT_FRAME_ZOOM_SAMPS	20
    #define ANIM_PARA_PAINT_FRAME_ROTATE_SAMPS	10
    #define ANIM_PARA_PAINT_FRAME_LAPSE_TIME	0.3f

    #define ANIM_SKETCH_BOOK_PAGE_FLIP_TIME	1.0f
    #define ANIM_SKETCH_BOOK_PAGE_FLIP_SAMPS	72


    /* Global data directory */
    const std::string ANIMDataDir();
};

#endif







