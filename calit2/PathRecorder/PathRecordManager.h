/***************************************************************
* File Name: PathRecordManager.h
*
* Class Name: PathRecordManager
*
***************************************************************/
#ifndef _PATH_RECORD_MANAGER_H_
#define _PATH_RECORD_MANAGER_H_


// C++
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Open scene graph
#include <osg/Matrixd>
#include <osg/Matrix>

// CalVR
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuText.h>

// Local include
#include "PathRecordEntryList.h"


using namespace std;
using namespace osg;
using namespace cvr;


/***************************************************************
* Class Name: PathRecordManager
***************************************************************/
class PathRecordManager
{
  public:
    PathRecordManager(const string &dataDir);

    /* define functional states */
    enum PathRecordState
    {
	SPARE,
	RECORD,
	PLAYBACK
    };

    void setState(const PathRecordState &stat) { mState = stat;	stop(); }
    void setPlaybackSpeed(const double &speed) { mPlaybackSpeed = speed; }
    void setActiveFileIdx(const int &idx);
    void updateTimer(const double &frameDur);
    void start();
    void pause();
    void stop();

    bool isPlayRecord() { return mPlayRecordFlag; }
    const int &getNumFiles() { return mNumFiles; }
    const int &getActiveFileIdx() { return mActiveFileIdx; }
    const string getFilename(const int &idx);
    const string timeToString(const double &t);
    double getTime() { return mTimer; }
    const PathRecordState &getState() { return mState; }
    MenuText *getInfoLabelPtr() { return mTimeInfoLabel; }

    void recordPathEntry(const double &scale, const Matrixd &xMat);
    bool playbackPathEntry(double &scale, Matrixd &xMat);

    enum PlaybackState
    {
        STARTED = 0,
        DONE
    };

    PlaybackState getPlaybackState() {return _currentPlaybackState;}

  protected:

    PlaybackState _currentPlaybackState;

    int mNumFiles, mActiveFileIdx;
    string mDataDir;
    FILE *mFilePtr;
    bool mPlayRecordFlag;
    double mPlaybackSpeed;
    double mTimer;

    PathRecordState mState;
    //coLabelMenuItem *mTimeInfoLabel;
    MenuText * mTimeInfoLabel;
    PathRecordEntryList *mRecordEntryListPtr;

    bool _smoothStart;
    bool _smoothStartActive;
    double _smoothStartTime;
    double _currentSmoothStartTime;
    osg::Matrix _smoothStartMat;
    double _smoothStartScale;
};

#endif
