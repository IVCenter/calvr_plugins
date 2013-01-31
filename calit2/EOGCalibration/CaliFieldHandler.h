/***************************************************************
* File Name: CaliFieldHandler.h
*
* Class Name: CaliFieldHandler
*
***************************************************************/
#ifndef _CALI_FIELD_HANDLER_H_
#define _CALI_FIELD_HANDLER_H_

// C++
#include <iostream>
#include <string.h>

// Open Scene Graph
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/ShapeDrawable>
#include <osg/Switch>

using namespace std;
using namespace osg;


/***************************************************************
*  Class: CaliFieldHandler
***************************************************************/
class CaliFieldHandler
{
  public:
    CaliFieldHandler(MatrixTransform *rootViewerTrans);

    void setVisible(bool flag);
    bool isVisible() { return mFlagVisible; }

    void updateWireFrames(const float left, const float right, const float up, const float down, const float depth);

  protected:
    void initWireFrames(MatrixTransform *rootViewerTrans);
    DrawElementsUInt *createEdgePrimitiveSet();

    /* osg geometries */
    bool mFlagVisible;
    Switch *mSwitch;
    Geode *mWireframeGeode;
    Geometry *mWireframeGeometry;

    /* calibration field parameters in rads */
    float gPhiMin, gPhiMax, gThetaMin, gThetaMax, gRadMin, gRadMax;    

    /* sampling parameter in integers */
    int gNumPhiSample, gNumThetaSample, gNumRadSample, gPhiSampleMin, gThetaSampleMin, gRadSampleMin,
	gPhiSampleMax, gThetaSampleMax, gRadSampleMax;

    /* static constant input parameters */
    static float RIGHT_RANGE_MAX, LEFT_RANGE_MAX, UPWARD_RANGE_MAX, DOWNWARD_RANGE_MAX, DEPTH_RANGE_MAX;
    static float RIGHT_RANGE_MIN, LEFT_RANGE_MIN, UPWARD_RANGE_MIN, DOWNWARD_RANGE_MIN, DEPTH_RANGE_MIN;
    static float PHI_RES, THETA_RES, RAD_RES;

    static inline void sphericToCartesian(const float &phi, const float &theta, const float &rad, Vec3 &pos)
    {
        pos.x() = rad * sin(theta) * cos(phi);
        pos.y() = rad * sin(theta) * sin(phi);
        pos.z() = rad * cos(theta);
    }
};


#endif
