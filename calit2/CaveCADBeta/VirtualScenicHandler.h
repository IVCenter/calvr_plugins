/***************************************************************
* File Name: VirtualScenicHandler.h
*
* Class Name: VirtualScenicHandler
*
***************************************************************/
#ifndef _VIRTUAL_SCENIC_HANDLER_H_
#define _VIRTUAL_SCENIC_HANDLER_H_


// C++
#include <math.h>
#include <string.h>
#include <iostream>

// Open scene graph
#include <osg/Group>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Program>
#include <osg/Shader>
#include <osg/StateSet>
#include <osg/Switch>
#include <osg/Texture2D>
#include <osg/Uniform>
#include <osgDB/ReadFile>

// Local includes
#include "AnimationModeler/ANIMRefXYPlane.h"
#include "AnimationModeler/ANIMRefSkyDome.h"
#include "AnimationModeler/ANIMRefWaterSurf.h"
#include "AnimationModeler/ANIMSketchBook.h"


/***************************************************************
* Class Name: VirtualScenicHandler
***************************************************************/
class VirtualScenicHandler
{
  public:
    VirtualScenicHandler(osg::Group* nonIntersectableSceneGraphPtr, osg::Group* intersectableSceneGraphPtr);

    /* visibility options */
    enum VisibilityOption
    {
	INVISIBLE,
	TRANSPARENT,
	SOLID
    };

    void setGeometryVisible(bool flag);
    void setSkyMaskingColorEnabled(bool flag);
    void setVSParamountPreviewHighlight(bool flag, osg::Geode *paintGeode);
    void setFloorplanPreviewHighlight(bool flag, osg::Geode *pageGeode);
    void switchVSParamount(const std::string &texname);
    void switchFloorplan(const int &idx, const VisibilityOption &option);
    void updateVSParameters(const osg::Matrixd &matShaderToWorld, const osg::Vec3 &sunDirWorld, 
			    const osg::Vec3 &viewDir, const osg::Vec3 &viewPos);
    void createFloorplanGeometry(const int numPages, CAVEAnimationModeler::ANIMPageEntry **pageEntryArray);

  protected:
    osg::Switch *mXYPlaneSwitch, *mSkyDomeSwitch, *mWaterSurfSwitch, *mFloorplanSwitch;
    osg::StateSet *mSkyDomeStateset, *mWatersurfStateset;
    osg::Geometry *mWatersurfGeometry;
    
    bool mWaterEnabled;
    int mFloorplanIdx;
    osg::Light *mSunLight, *mPointLight;
    osg::LightSource *mSunLightSource, *mPointLightSource;

    /* Two light sources are created for design object handler scene graph:
       'createSunLight' creates adaptive sun light for all intersectable objects,
       'createPointLight' creates static light for non-intersectable objects.
    */
    osg::Group *createSunLight(osg::StateSet *stateset);
    osg::Group *createPointLight(osg::StateSet *stateset);

    void interpretColorParameters(const osg::Vec3 &sunDir, osg::Vec4 &sunColor,
				osg::Vec4 &skyColor, osg::Vec4 &skyFadingColor);
};

#endif

