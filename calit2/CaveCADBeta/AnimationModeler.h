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
#include "Geometry/CAVEGeodeIcon.h"
#include "Geometry/CAVEGeodeReference.h"
#include "Geometry/CAVEGeodeSnapWireframe.h"
#include "Geometry/CAVEGeodeSnapSolidshape.h"
#include "Geometry/CAVEGeodeShape.h"
#include "Geometry/CAVEGroupReference.h"


namespace CAVEAnimationModeler
{

    #define ANIM_VIRTUAL_SPHERE_RADIUS		0.2
    #define ANIM_VIRTUAL_SEASONS_MAP_RADIUS	0.25
    #define ANIM_VIRTUAL_SEASONS_MAP_ALPHA	0.8
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

    #define ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_TIME	1.0f
    #define ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_SAMPS	36

    typedef std::list<osgParticle::Emitter*> ANIMEmitterList;

    // Data directory, earth light direction
    const std::string ANIMDataDir();
    const osg::Vec3 ANIMVirtualEarthLightDir();

    /***************************************************************
    * Function: Animation & model prototypes
    ***************************************************************/
    osg::MatrixTransform* ANIMCreateDesignStateParticleSystem(ANIMEmitterList *dsEmitterList);


    // Virtual Sphere

    void ANIMCreateVirtualSphere(osg::PositionAttitudeTransform** xformScaleFwd, 
                                 osg::PositionAttitudeTransform** xformScaleBwd);
    osg::MatrixTransform* ANIMCreateVirtualSpherePointSprite();


    // Design Objects

    osg::MatrixTransform* ANIMCreateRefXYPlane();
    osg::MatrixTransform* ANIMCreateRefSkyDome(osg::StateSet **stateset);
    osg::MatrixTransform* ANIMCreateRefWaterSurf(osg::StateSet **stateset,
                                                 osg::Geometry **watersurfGeometry);
    osg::Geode* ANIMCreateWiredSphereGeode(const int numLatiSegs, const int numLongiSegs,
					                       const float rad, const osg::Vec4 color);


    // Virtual Earth

    void ANIMLoadVirtualEarthReferenceLevel(osg::Switch **designStateSwitch, osg::Geode **seasonsMapNode);
    void ANIMLoadVirtualEarthEclipticLevel(osg::Switch **eclipticSwitch);
    void ANIMLoadVirtualEarthEquatorLevel(osg::Switch **equatorSwitch, osg::Geode **earthGeode,
		osg::PositionAttitudeTransform **xformTransFwd, osg::PositionAttitudeTransform **xformTransBwd,
		osg::MatrixTransform **fixedPinIndicatorTrans, osg::MatrixTransform **fixedPinTrans);


    // Viewpoints

    void ANIMCreateViewpoints(std::vector<osg::PositionAttitudeTransform*>* fwdVec,
                              std::vector<osg::PositionAttitudeTransform*>* bwdVec);
    void ANIMAddViewpoint(std::vector<osg::PositionAttitudeTransform*>* fwdVec,
                          std::vector<osg::PositionAttitudeTransform*>* bwdVec);


    // Panoramas

    /***************************************************************
    * Class: ANIMParamountSwitchEntry
    ***************************************************************/
    class ANIMParamountSwitchEntry
    {
      public:
        osg::MatrixTransform *mMatrixTrans;
        osg::Switch *mSwitch;
        osg::AnimationPathCallback *mZoomInAnim, *mZoomOutAnim;
        osg::Geode *mPaintGeode;
        std::string mTexFilename;
    };

    void ANIMLoadParamountPaintFrames(osg::PositionAttitudeTransform** xformScaleFwd,
                                      osg::PositionAttitudeTransform** xformScaleBwd,
                                      int &numParas, float &paraswitchRadius,
                                      ANIMParamountSwitchEntry ***paraEntryArray);
    void ANIMCreateParamountPaintFrameAnimation(osg::AnimationPathCallback **zoomInCallback,
						osg::AnimationPathCallback **zoomOutCallback);
    osg::Geode *ANIMCreateParamountPaintGeode(const std::string &texfilename);


    // Sketchbook

    /***************************************************************
    * Class: ANIMPageEntry
    ***************************************************************/
    class ANIMPageEntry
    {
      public:
        osg::Switch *mSwitch;
        osg::AnimationPathCallback *mFlipUpAnim, *mFlipDownAnim;
        osg::Geode *mPageGeode;
        std::string mTexFilename;
        float mLength, mWidth, mAlti;
    };

    void ANIMLoadSketchBook(osg::PositionAttitudeTransform** xformScaleFwd,
			    osg::PositionAttitudeTransform** xformScaleBwd,
			    int &numPages, ANIMPageEntry ***pageEntryArray);
    void ANIMCreateSinglePageGeodeAnimation(const std::string& texfilename,
					    osg::Geode **flipUpGeode, osg::Geode **flipDownGeode,
					    osg::AnimationPathCallback **flipUpCallback,
					    osg::AnimationPathCallback **flipDownCallback);


    // Geometry creator

    /***************************************************************
    * Class: ANIMShapeSwitchEntry
    ***************************************************************/
    class ANIMShapeSwitchEntry
    {
      public:
        osg::Switch *mSwitch;
        osg::AnimationPathCallback *mFlipUpFwdAnim, *mFlipDownFwdAnim;
        osg::AnimationPathCallback *mFlipUpBwdAnim, *mFlipDownBwdAnim;
    };

    void ANIMLoadGeometryCreator(osg::PositionAttitudeTransform** xformScaleFwd,
		     	 osg::PositionAttitudeTransform** xformScaleBwd,
				 osg::Switch **sphereExteriorSwitch, osg::Geode **sphereExteriorGeode,
				 int &numTypes, ANIMShapeSwitchEntry ***shapeSwitchEntryArray);
    void ANIMCreateSingleShapeSwitchAnimation(ANIMShapeSwitchEntry **shapeEntry, const CAVEGeodeShape::Type &typ);
    void ANIMLoadObjectHandler(osg::Switch **snapWireframeSwitch, osg::Switch **snapSolidshapeSwitch);


    // Object Placer
    void ANIMCreateObjectPlacer(std::vector<osg::PositionAttitudeTransform*>* fwdVec, 
                                std::vector<osg::PositionAttitudeTransform*>* bwdVec);

};

#endif

