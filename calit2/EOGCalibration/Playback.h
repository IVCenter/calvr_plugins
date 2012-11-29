/***************************************************************
* File Name: Playback.h
*
* Class Name: Playback
*
***************************************************************/
#ifndef _PLAY_BACK_H_
#define _PLAY_BACK_H_


// C++
#include <iostream>
#include <string.h>
#include <vector>

// Open Scene Graph
#include <osg/Vec3>
#include <osg/Vec4>

using namespace std;
using namespace osg;


/***************************************************************
*  Class: PlaybackEntry
***************************************************************/
class PlaybackEntry
{
  public:
	double mTS;					// time stamp
	osg::Vec3 mHeadPos;			// head position
	osg::Vec3 mCaliBallPos;		// stimuli calibration ball position
	osg::Vec3 mPredBallPos;		// predictive ball position
};

typedef std::vector<PlaybackEntry*> PlaybackEntryVector;

#endif
