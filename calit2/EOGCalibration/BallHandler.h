/***************************************************************
* File Name: BallHandler.h
*
* Class Name: BallHandler
*
***************************************************************/
#ifndef _BALL_HANDLER_H_
#define _BALL_HANDLER_H_

// C++
#include <fstream>
#include <iostream>
#include <string.h>

// Open Scene Graph
#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/ShapeDrawable>
#include <osg/Switch>
#include <osgDB/ReadFile>

#include <cvrConfig/ConfigManager.h>

// Local
#include "Playback.h"


using namespace std;
using namespace osg;


/***************************************************************
*  Class: BallHandler
***************************************************************/
class BallHandler
{
  public:
    BallHandler();

    virtual void setVisible(bool flag);
    virtual bool isVisible() { return mFlagVisible; }

    virtual void updateCaliBall(const float &phi, const float &theta, const float &rad);

	static void setDataDir(const std::string &datadir) { gDataDir = datadir; }

  protected:
    void initCaliBallGeometry(MatrixTransform *rootViewerTrans, const Vec4 &exteriorBallColor,
        const Vec4 &interiorBallColor = osg::Vec4(1.0, 1.0, 1.0, 1.0));

    /* osg geometries */
    bool mFlagVisible;
    Switch *mBallSwitch;
    MatrixTransform *mBallTrans;
    Geode *mBoundingBallGeode, *mCenterBallGeode;

    /* calibration ball parameters */
    //static float BOUNDING_BALL_SIZE;
    //static float CENTER_BALL_SIZE;
    float BOUNDING_BALL_SIZE;
    float CENTER_BALL_SIZE;

	static std::string gDataDir;

    static inline void sphericToCartetion(const float &phi, const float &theta, const float &rad, Vec3 &pos)
    {
		pos.x() = rad * sin(theta) * cos(phi);
		pos.y() = rad * sin(theta) * sin(phi);
		pos.z() = rad * cos(theta);
    }
};


/***************************************************************
*  Class: CaliBallHandler
***************************************************************/
class CaliBallHandler: public BallHandler
{
  public:
	CaliBallHandler(MatrixTransform *rootViewerTrans);
};


/***************************************************************
*  Class: PlaybackBallHandler
***************************************************************/
class PlaybackBallHandler: public BallHandler
{
  public:
	PlaybackBallHandler(MatrixTransform *rootViewerTrans);

	/* function called by 'CalibrationController'*/
	void importPlaybackFile(const std::string &filename);
	void resertVirtualTimer() { mVirtualTimer = 0.0; }
	void updatePlaybackTime(const double &frameDuration);
	void updatePlaybackBallPos();

	const std::string getPlaybackTimeLabel();

	virtual void setVisible(bool flag);

  protected:

	/* virtual playback time: it is paused if playback ball is not visible */
	int mPlaybackItr;
	double mVirtualTimer;
	double mStartVirtualTime, mMaxVirtualTime;
	PlaybackEntryVector mEntryVector;

	/* additional geometries */
	Switch *mHeadSwitch;
	Node *mEyeBallNode;
	MatrixTransform *mHeadTrans, *mPoleTrans, *mEyeBallTrans, *mGhostBallTrans;
	Geode *mPoleGeode;

	static float VISUAL_POLE_RADIUS;
	static float VISUAL_POLE_LENGTH;
};

#endif






