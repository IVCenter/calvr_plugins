/***************************************************************
* File Name: CAVEGroupReference.h
*
* Description: Derived class from CAVEGroup
*
***************************************************************/

#ifndef _CAVE_GROUP_REFERENCE_H_
#define _CAVE_GROUP_REFERENCE_H_


// C++
#include <iostream>
#include <list>
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
#include "CAVEGeodeReference.h"
#include "CAVEGroup.h"


/***************************************************************
* Class: CAVEGroupReference
***************************************************************/
class CAVEGroupReference: public CAVEGroup
{
  public:
    CAVEGroupReference();

    virtual void setCenterPos(const osg::Vec3 &center);

    /* Set front direction of the input device. This vector is
       proejcted onto XY plane first and can be used to align
       all reference geometries with viewer's front. */
    static void setPointerDir(const osg::Vec3 &pointerDir);

  protected:
    osg::Vec3 mCenter;
    osg::MatrixTransform *mMatrixTrans;

    static osg::Vec3 gPointerDir;
};


/***************************************************************
* Class: CAVEGroupReferenceAxis
***************************************************************/
class CAVEGroupReferenceAxis: public CAVEGroupReference
{
  public:
    CAVEGroupReferenceAxis();

    void setAxisMasking(bool flag);
    void setTextEnabled(bool flag) { mTextEnabledFlag = flag; }
    void updateUnitGridSizeInfo(const std::string &infoStr);
    void updateDiagonal(const osg::Vec3 &wireFrameVect, const osg::Vec3 &solidShapeVect);
    bool isVisible() { return mVisibleFlag; }

  protected:
    bool mVisibleFlag, mTextEnabledFlag;
    osg::Switch *mAxisSwitch;
    osg::MatrixTransform *mXDirTrans, *mYDirTrans, *mZDirTrans;
    CAVEGeodeReferenceAxis *mXAxisGeode, *mYAxisGeode, *mZAxisGeode;

    /* 3D text associated objects */
    osg::MatrixTransform *mGridInfoTextTrans, *mXAxisTextTrans, *mYAxisTextTrans, *mZAxisTextTrans;
    osgText::Text3D *mGridInfoText, *mXAxisText, *mYAxisText, *mZAxisText;

    osg::Geode *createText3D(osgText::Text3D **text);

    static const float gCharSize;
    static const float gCharDepth;
};


/***************************************************************
* Class: CAVEGroupReferencePlane
***************************************************************/
class CAVEGroupReferencePlane: public CAVEGroupReference
{
  public:
    CAVEGroupReferencePlane();

    virtual void setCenterPos(const osg::Vec3 &center);

    void setUnitGridSize(const float &gridsize);
    void setPlaneMasking(bool flagXY, bool flagXZ, bool flagYZ);

  protected:
    float mUnitGridSize;

    osg::Switch *mXYPlaneSwitch, *mXZPlaneSwitch, *mYZPlaneSwitch;
    osg::MatrixTransform *mXYPlaneTrans, *mXZPlaneTrans, *mYZPlaneTrans;
    CAVEGeodeReferencePlane *mXYPlaneGeode, *mXZPlaneGeode, *mYZPlaneGeode;
};



#endif



