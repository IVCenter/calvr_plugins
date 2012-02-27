/***************************************************************
* File Name: CalibrationController.h
*
* Class Name: CalibrationController
*
***************************************************************/
#ifndef _CALIBRATION_CONTROLLER_H_
#define _CALIBRATION_CONTROLLER_H_

// C++
#include <iostream>
#include <string.h>
#include <iomanip>
#include <fstream>
#include <time.h>

// Open Scene Graph
#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/ShapeDrawable>
#include <osg/Switch>

// Local
#include "BallHandler.h"
#include "CaliFieldHandler.h"

using namespace std;
using namespace osg;

/***************************************************************
*  Class: CalibrationController
***************************************************************/
class CalibrationController
{
  public:
    CalibrationController(Group *rootGroup, const string &datadir);

    /* types of calibration parameters */
    enum CaliFieldParameter
    {
	LEFT_RANGE,
	RIGHT_RANGE,
	UPWARD_RANGE,
	DOWNWARD_RANGE,
	DEPTH_RANGE,

	HORIZONTAL_FREQ,
	VERTICAL_FREQ,
	DEPTH_FREQ
    };

    /* OpenVRUI control functions */
    void startCalibration();
    void stopCalibration();
	void startPlayback();
	void stopPlayback();
    bool isCalibrationStarted() { return mCaliFlag; }
	bool isPlaybackStarted() { return mPlaybackFlag; }
    void setCaliBallVisible(bool flag);
    bool isCaliBallVisible();
	void setPlaybackBallVisible(bool flag);
    bool isPlaybackBallVisible();
    void setCaliFieldVisible(bool flag);
    bool isCaliFieldVisible();
    void resetCaliBall();
    void resetCaliField(const Matrixf &invBaseMat);

    /* update functions: 1) System timer; 2) Preframe rendering; 3) UI listener. */
    void updateCaliTime(const double &frameDuration);
    void updateCaliBallPos(float &phi, float &theta, float &rad);
    void updateCaliParam(const CaliFieldParameter& typ, const float &val);
	void updatePlaybackTime(const double &frameDuration);
	void updatePlaybackBallPos();
    void updateViewMat(const Matrixd &viewMat) { mViewMat = viewMat; }

	const std::string getPlaybackTimeLabel() { return mPlaybackBallHandler->getPlaybackTimeLabel(); }

  protected:
    bool mCaliFlag, mPlaybackFlag;
    double mTimer;
    MatrixTransform *mViewerAlignmentTrans, *mNoseOffsetTrans;

    CaliBallHandler *mCaliBallHandler;
	PlaybackBallHandler *mPlaybackBallHandler;
    CaliFieldHandler *mCaliFieldHandler;

    string mDataDir;
    Matrixd mViewMat;	// update with viewer's matrix

    /* calibration field parameters in degrees */
    float mLeftRange, mRightRange, mUpwardRange, mDownwardRange, mMinDepthRange, mMaxDepthRange, 
	  mHorFreq, mVerFreq, mDepFreq;
    Vec3 mFieldRight, mFieldFront, mFieldUp, mFieldPos;

    /* offset distance that shift center of calibration field from head to nose */
    static const float gNoseOffset;
};


#endif
