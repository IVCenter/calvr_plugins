/***************************************************************
* File Name: CAVEGroupEditWireframe.h
*
* Description: Derived class from CAVEGroup, decendent of
* 'CAVEGroupIconSurface' as container 'CAVEGroupEditWireframe' 
* objects under 'DesignObjectHandler::mCAVEGeodeIconSurfaceSwitch'
*
***************************************************************/

#ifndef _CAVE_GROUP_EDIT_WIREFRAME_H_
#define _CAVE_GROUP_EDIT_WIREFRAME_H_


// C++
#include <iostream>
#include <list>
#include <string>
#include <vector>
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
#include "CAVEGeometry.h"
#include "CAVEGeodeIcon.h"
#include "CAVEGeodeShape.h"
#include "CAVEGeodeEditWireframe.h"
#include "CAVEGroup.h"


class CAVEGroupEditGeodeWireframe;
class CAVEGroupEditGeometryWireframe;

/* define two types of wireframe vectors that used in 'DOGeometryCollector' */
typedef std::vector<CAVEGroupEditGeodeWireframe*>	CAVEGroupEditGeodeWireframeVector;
typedef std::vector<CAVEGroupEditGeometryWireframe*>	CAVEGroupEditGeometryWireframeVector;
typedef std::vector<osg::MatrixTransform*> MatrixTransVector;


/***************************************************************
* Class: CAVEGroupEditWireframe
***************************************************************/
class CAVEGroupEditWireframe: public CAVEGroup
{
  public:

    CAVEGroupEditWireframe();

    void setPrimaryFlag(bool flag) { mPrimaryFlag = flag; }
    void resetInfoOrientation();

    /* functions called by 'DOGeometryEditor' during snapping */
    virtual void applyTranslation(const osg::Vec3s &gridSVect, const float &gridUnitLegnth, 
					const std::string &gridUnitLegnthInfo) = 0;
    virtual void applyRotation(const osg::Vec3s &axisSVect, const float &gridUnitAngle, 
					const std::string &gridUnitAngleInfo) = 0;
    virtual void applyScaling(const short &nOffsetSegs, const osg::Vec3 &gridUnitScaleVect, 
					const std::string &gridUnitScaleInfo) = 0;

    virtual void updateGridUnits(const float &gridUnitLegnth, const float &gridUnitAngle, const float &gridUnitScale) = 0;

    /* functions called by 'DOGeometryEditor' when snapping is finished */
    virtual void applyEditorInfo(CAVEGeodeShape::EditorInfo **infoPtr) = 0;
    virtual void clearActiveWireframes() = 0;

    /* Set front direction of the input device. This vector is
       proejcted onto XY plane first and can be used to align
       all reference geometries with viewer's front. */
    static void setPointerDir(const osg::Vec3 &pointerDir);

  protected:

    /* 'mPrimaryFlag' is served as indicator of whether turning on the text and rotational wireframes through
        editting process. */
    bool mPrimaryFlag;

    /* scaling matrix for bounding box and bounding sphere */
    osg::Matrixd mBoundBoxScaleMat, mBoundSphereScaleMat;
    float mBoundingRadius;

    /* short integer vectors that take record of current values of 'gridIntVect' and 'axisIntVect' */
    osg::Vec3s mMoveSVect, mRotateSVect;

    /* use a short integer and a vec3f to record scaling parameters */
    short mScaleNumSegs;		// can be either positive or negative, depending on scaling direction
    osg::Vec3f mScaleUnitVect;		// always with positive values

    /* osg text objects showing editting information */
    osg::Switch *mEditInfoTextSwitch;
    osg::MatrixTransform *mEditInfoTextTrans;
    osgText::Text3D *mEditInfoText;
    osg::Geode *createText3D(osgText::Text3D **text);

    /* osg text size parameters */
    static const float gCharSize;
    static const float gCharDepth;

    static osg::Vec3 gPointerDir;
};


/***************************************************************
* Class: CAVEGroupEditGeodeWireframe
***************************************************************/
class CAVEGroupEditGeodeWireframe: public CAVEGroupEditWireframe
{
  public:
    CAVEGroupEditGeodeWireframe();

    /* function called by 'ANIMGeometryCollector' and 'DOGeometryEditor' */
    void acceptCAVEGeodeShape(CAVEGeodeShape *shapeGeode, const osg::Vec3 &refShapeCenter);
    void updateCAVEGeodeShape(CAVEGeodeShape *shapeGeode);

    /* functions called by 'DOGeometryEditor' during snapping */
    virtual void applyTranslation(const osg::Vec3s &gridSVect, const float &gridUnitLegnth, 
					const std::string &gridUnitLegnthInfo);
    virtual void applyRotation(const osg::Vec3s &axisSVect, const float &gridUnitAngle, 
					const std::string &gridUnitAngleInfo);
    virtual void applyScaling(const short &nOffsetSegs, const osg::Vec3 &gridUnitScaleVect, 
					const std::string &gridUnitScaleInfo);

    virtual void updateGridUnits(const float &gridUnitLegnth, const float &gridUnitAngle, const float &gridUnitScale);

    /* functions called by 'DOGeometryEditor' when snapping is finished */
    virtual void applyEditorInfo(CAVEGeodeShape::EditorInfo **infoPtr);
    virtual void clearActiveWireframes();

  protected:

    /* switches associated with different types of operations, for each type of operation,
       the swich contains a series of MatrixTransform objects with different offsets, these
       MatrixTransform objects will be pushed back into the following MatrixTransVector */
    osg::Switch *mMoveSwitch, *mRotateSwitch, *mManipulateSwitch;
    MatrixTransVector mMoveMatTransVector, mRotateMatTransVector, mManipulateMatTransVector;

    /* accumulated root level transformation, is a combination of translations and rotations, this matrix is
       updated in 'applyEditorInfo' at the end of each snapping operation */
    osg::Matrixd mAccRootMat;

    /* reference shape center: central position of the first selected 'CAVEGeodeShape' in world space, this vector
       is with the same value for all geode wireframe objects before all selected 'CAVEGeodeShape' are released. */
    osg::Vec3 mRefShapeCenter;
};


/***************************************************************
* Class: CAVEGroupEditGeometryWireframe
***************************************************************/
class CAVEGroupEditGeometryWireframe: public CAVEGroupEditWireframe
{
  public:
    CAVEGroupEditGeometryWireframe();

    /* function called by 'ANIMGeometryCollector' and 'DOGeometryEditor' */
    void acceptCAVEGeometry(CAVEGeometry *geometry, const osg::Vec3 &refShapeCenter);
    void updateCAVEGeometry();

    /* functions called by 'DOGeometryEditor' during snapping */
    virtual void applyTranslation(const osg::Vec3s &gridSVect, const float &gridUnitLegnth, 
					const std::string &gridUnitLegnthInfo);
    virtual void applyRotation(const osg::Vec3s &axisSVect, const float &gridUnitAngle, 
					const std::string &gridUnitAngleInfo);
    virtual void applyScaling(const short &nOffsetSegs, const osg::Vec3 &gridUnitScaleVect, 
					const std::string &gridUnitScaleInfo);

    virtual void updateGridUnits(const float &gridUnitLegnth, const float &gridUnitAngle, const float &gridUnitScale);

    /* functions called by 'DOGeometryEditor' when snapping is finished */
    virtual void applyEditorInfo(CAVEGeodeShape::EditorInfo **infoPtr);
    virtual void clearActiveWireframes();

  protected:
    /* reference 'CAVEGeodeShape' center in world space, only assigned in 'acceptCAVEGeometry' */
    osg::Vec3 mRefShapeCenter;

    /* Similar as 'CAVEGroupEditGeodeWireframe', 'mSwitch' contains a set of MatrixTransform objects 
       with interpolated offsets, either in forms of translation, scaling or rotation. */
    osg::MatrixTransform *mRootTrans;
    osg::Switch *mSwitch;
    MatrixTransVector mMatTransVector;

    /* geometry pointer used to track the origin CAVEGeometry that to be editted */
    CAVEGeometry *mGeometryReferencePtr;
};


#endif













