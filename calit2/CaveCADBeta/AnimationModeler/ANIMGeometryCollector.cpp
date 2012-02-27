/***************************************************************
* Animation File Name: ANIMGeometryCollector.cpp
*
* Description: Load geometry collector shapes & objects
*
* Written by ZHANG Lelin on Jan 25, 2011
*
***************************************************************/
#include "ANIMGeometryCollector.h"

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{

/***************************************************************
* Function: ANIMLoadGeometryCollectorIconSurfaces()
*
* 'shapeGeode': 'CAVEGeodeShape' collected in world space
* 'shapeGeodeRef': ghost 'CAVEGeodeShape' as editing indicator
*
***************************************************************/
void ANIMLoadGeometryCollectorIconSurfaces(PositionAttitudeTransform** surfacesPATrans,
		CAVEGroupIconSurface **iconSurfaceGroup, CAVEGeodeShape *shapeGeode, CAVEGeodeShape *shapeGeodeRef)
{
    *surfacesPATrans = new PositionAttitudeTransform;

    /* create an instance of 'CAVEGroupIconSurface', add generated 'CAVEGeodeIconSurface' objects
       into its 'mCAVEGeodeIconVector' fields. */
    *iconSurfaceGroup = new CAVEGroupIconSurface();
    (*iconSurfaceGroup)->acceptCAVEGeodeShape(shapeGeode, shapeGeodeRef);
    (*surfacesPATrans)->addChild(*iconSurfaceGroup);

    /* create pickup animation callback for 'surfacesPATrans' */
    AnimationPath* surfacesAnim = new AnimationPath;
    surfacesAnim->setLoopMode(AnimationPath::NO_LOOPING);

    /* get translation & scaling hints from 'CAVEGroupIconSurface' */
    Vec3 shapeCenter, iconCenter;
    CAVEGroupIconSurface::getSurfaceTranslationHint(shapeCenter, iconCenter);
    const float scaleVal = CAVEGroupIconSurface::getSurfaceScalingHint();

    Vec3 scaleFwd, transFwd;
    float step = 1.f / ANIM_GEOMETRY_COLLECTOR_SURFACE_PICKUP_SAMPS;
    for (int i = 0; i < ANIM_GEOMETRY_COLLECTOR_SURFACE_PICKUP_SAMPS + 1; i++)
    {
	float val = i * step;
	float scale = (scaleVal - 1.f) * val + 1.f; 
	scaleFwd = Vec3(scale, scale, scale);
	transFwd = shapeCenter + (iconCenter - shapeCenter) * val;
	surfacesAnim->insert(val, AnimationPath::ControlPoint(transFwd, Quat(), scaleFwd));
    }

    AnimationPathCallback *surfacesAnimCallback = new AnimationPathCallback(surfacesAnim, 
				0.0, 1.f / ANIM_GEOMETRY_COLLECTOR_SURFACE_PICKUP_TIME);
    (*surfacesPATrans)->setUpdateCallback(surfacesAnimCallback);
}


/***************************************************************
* Function: ANIMLoadGeometryCollectorGeodeWireframe()
***************************************************************/
void ANIMLoadGeometryCollectorGeodeWireframe(MatrixTransform **wireframeTrans, 
		CAVEGroupEditGeodeWireframe **editGeodeWireframe, CAVEGeodeShape *shapeGeode)
{
    *wireframeTrans = new MatrixTransform();
    *editGeodeWireframe = new CAVEGroupEditGeodeWireframe;

    /* get translation & scaling hints from 'CAVEGroupIconSurface' */
    Vec3 shapeCenter, iconCenter;
    CAVEGroupIconSurface::getSurfaceTranslationHint(shapeCenter, iconCenter);
    const float scaleVal = CAVEGroupIconSurface::getSurfaceScalingHint();

    (*editGeodeWireframe)->acceptCAVEGeodeShape(shapeGeode, shapeCenter);

    Matrixd scaleMat, transMat;
    scaleMat.makeScale(Vec3(scaleVal, scaleVal, scaleVal));
    transMat.makeTranslate(iconCenter);
    (*wireframeTrans)->setMatrix(scaleMat * transMat);
    (*wireframeTrans)->addChild(*editGeodeWireframe);
}


/***************************************************************
* Function: ANIMLoadGeometryCollectorGeometryWireframe()
***************************************************************/
void ANIMLoadGeometryCollectorGeometryWireframe(MatrixTransform **wireframeTrans, 
		CAVEGroupEditGeometryWireframe **editGeometryWireframe, CAVEGeometry *geometry)
{
    *wireframeTrans = new MatrixTransform();
    *editGeometryWireframe = new CAVEGroupEditGeometryWireframe;

    /* get translation & scaling hints from 'CAVEGroupIconSurface' */
    Vec3 shapeCenter, iconCenter;
    CAVEGroupIconSurface::getSurfaceTranslationHint(shapeCenter, iconCenter);
    const float scaleVal = CAVEGroupIconSurface::getSurfaceScalingHint();

    (*editGeometryWireframe)->acceptCAVEGeometry(geometry, shapeCenter);

    Matrixd scaleMat, transMat;
    scaleMat.makeScale(Vec3(scaleVal, scaleVal, scaleVal));
    transMat.makeTranslate(iconCenter);
    (*wireframeTrans)->setMatrix(scaleMat * transMat);
    (*wireframeTrans)->addChild(*editGeometryWireframe);
}


};








