/***************************************************************
* File Name: PathRecordManager.cpp
*
* Description: Read/Write record entry files and perform path
* recordings and playbacks.
*
* Written by ZHANG Lelin on Oct 15, 2010
*
***************************************************************/
#include "PathRecordManager.h"

#include <cvrKernel/PluginHelper.h>

using namespace std;

// Constructor
PathRecordManager::PathRecordManager(const string &dataDir): mNumFiles(12), mActiveFileIdx(0), 
		mFilePtr(NULL), mPlayRecordFlag(false), mPlaybackSpeed(1.0f), mTimer(0.0), mState(RECORD), _smoothStart(false),
		_smoothStartTime(3.0), _currentPlaybackState(DONE)
{
    mDataDir = dataDir;

    //mTimeInfoLabel = new coLabelMenuItem("");
    //mTimeInfoLabel->setLabel("   00:00:000 / 00:00:000");
    mTimeInfoLabel = new MenuText("   00:00:000 / 00:00:000");

    mRecordEntryListPtr = new PathRecordEntryList();
}


/***************************************************************
*  Function: setActiveFileIdx()
***************************************************************/
void PathRecordManager::setActiveFileIdx(const int &idx)
{
    mActiveFileIdx = idx;
    stop();

    if (idx < 0) 
    {
	mTimeInfoLabel->setText("   No path file has been loaded.");
	return;
    }
}


/***************************************************************
*  Function: updateTimer()
*
*  Record/Playback share the same timer, which is updated only
*  when paths are being recorded or played back
*
***************************************************************/
void PathRecordManager::updateTimer(const double &frameDur)
{
    if (!mPlayRecordFlag) return;
    //if(!_smoothStart)
    {
	mTimer += frameDur * mPlaybackSpeed;
    }

    if (mState == RECORD)
    {
	string leftStr = timeToString(mTimer);
	mTimeInfoLabel->setText(string("   ") + leftStr + " / "  + leftStr);
    } else {
	//if(_smoothStart)
	//{
	//    _currentSmoothStartTime -= frameDur;
	//}
	//else
	{
	    double period = mRecordEntryListPtr->getPeriodical();
	    double clamptime = mTimer - (int)(mTimer / period) * period;

	    if (mTimer > period)
	    {
		stop();
		return;
	    }

	    string leftStr = timeToString(clamptime);
	    string rightStr = timeToString(period);
	    mTimeInfoLabel->setText(string("   ") + leftStr + " / "  + rightStr);
	}
    }
}


/***************************************************************
*  Function: menuEvent()
***************************************************************/
void PathRecordManager::start()
{
    if (mActiveFileIdx < 0) return;
    mPlayRecordFlag = true;

    /* open active file, load record entries to memory if there is any */
    string filename = mDataDir + "/" + getFilename(mActiveFileIdx);

    if (mState == RECORD)
    { 
	mFilePtr = fopen(filename.c_str(), "wb");
	if (!mFilePtr) cerr << "ERROR: PathRecordManager: Can't open file " << filename << endl;
    }
    else if (mState == PLAYBACK) 
    {
        _currentPlaybackState = STARTED;
	mFilePtr = fopen(filename.c_str(), "rb");
	if (!mFilePtr) cerr << "ERROR: PathRecordManager: Can't open file " << filename << endl;
	else mRecordEntryListPtr->loadFile(mFilePtr);

	//_smoothStart = true;
	//_currentSmoothStartTime = _smoothStartTime;
	//_smoothStartMat = PluginHelper::getObjectMatrix();
	//_smoothStartScale = PluginHelper::getObjectScale();
    }
}


/***************************************************************
*  Function: menuEvent()
***************************************************************/
void PathRecordManager::pause()
{
    mPlayRecordFlag = false;
}


/***************************************************************
*  Function: menuEvent()
***************************************************************/
void PathRecordManager::stop()
{
    _currentPlaybackState = DONE;
    mTimer = 0.0;
    mPlayRecordFlag = false;
    mTimeInfoLabel->setText("   00:00:000 / 00:00:000");

    if(mFilePtr) fclose((FILE*)mFilePtr);
    mFilePtr = NULL;
}


/***************************************************************
*  Function: getFilename()
*
*  Get record file name given its index number
*
***************************************************************/
const string PathRecordManager::getFilename(const int &idx)
{
    string filename = string("");

    if (idx >= 0 && idx < mNumFiles)
    {
	char suffix[8];
	sprintf(suffix, "%d", idx);

	if (idx < 10) filename = filename + "Path00" + suffix + ".rcd";
	else if (idx < 100) filename = filename + "Path0" + suffix + ".rcd";
	else if (idx < 1000) filename = filename + "Path" + suffix + ".rcd";
    }

    return filename;
}


/***************************************************************
*  Function: timeToString()
*
*  Convert timer value to a string that can be post on board
*
***************************************************************/
const string PathRecordManager::timeToString(const double &t)
{
    char strmin[16], strsec[16], strmsec[16];
    int min, sec, msec;

    min = (int)(t / 60);
    sec = (int)(t - min * 60);
    msec = (int)((t - (int) t) * 1000);

    if (min >= 100) min = min - (int)(min / 100) * 100;
    if (min < 10) sprintf(strmin, "0%d:", min);
    else sprintf(strmin, "%d:", min);

    if (sec < 10) sprintf(strsec, "0%d:", sec);
    else sprintf(strsec, "%d:", sec);

    if (msec < 10) sprintf(strmsec, "00%d", msec);
    if (msec < 100) sprintf(strmsec, "0%d", msec);
    else sprintf(strmsec, "%d", msec);

    return string(strmin) + string(strsec) + string(strmsec);
}


/***************************************************************
*  Function: recordPathEntry()
*
*  Write current scale and xform matrix to file with time stamp
*
***************************************************************/
void PathRecordManager::recordPathEntry(const double &scale, const Matrixd &xMat)
{
    if (!mFilePtr) return;

    fwrite(&mTimer,sizeof(double),1,mFilePtr);

    double scaled = scale;

    fwrite(&scaled,sizeof(double),1,mFilePtr);
    fwrite(xMat.ptr(),16*sizeof(double),1,mFilePtr);
    /*fprintf(mFilePtr, "%f %f ", mTimer, scale);
    fprintf(mFilePtr, "%f %f %f %f ", xMat(0, 0), xMat(0, 1), xMat(0, 2), xMat(0, 3));
    fprintf(mFilePtr, "%f %f %f %f ", xMat(1, 0), xMat(1, 1), xMat(1, 2), xMat(1, 3));
    fprintf(mFilePtr, "%f %f %f %f ", xMat(2, 0), xMat(2, 1), xMat(2, 2), xMat(2, 3));
    fprintf(mFilePtr, "%f %f %f %f ", xMat(3, 0), xMat(3, 1), xMat(3, 2), xMat(3, 3));
    fprintf(mFilePtr, "\n");*/
}


/***************************************************************
*  Function: playbackPathEntry()
*
*  Get interpolated scale values and xfomr matrix with respect
*  to current time stamp
*
***************************************************************/
bool PathRecordManager::playbackPathEntry(double &scale, Matrixd &xMat)
{
    /*if(_smoothStart)
    {
	osg::Matrix nmat;
	double nscale;
	if(mTimer != 0)
	{
	    mRecordEntryListPtr->lookupRecordEntry(mTimer, nscale, nmat);
	}
	else
	{
            mTimer = 0.1;
	    mRecordEntryListPtr->lookupRecordEntry(mTimer, nscale, nmat);
	}
	if(_currentSmoothStartTime < 0.0)
	{
	    _currentSmoothStartTime = 0.0;
	    _smoothStart = false;
	}

	double fact = _currentSmoothStartTime / _smoothStartTime;
	for(int i = 0; i < 16; i++)
	{
	    xMat.ptr()[i] = fact * _smoothStartMat.ptr()[i] + (1.0 -fact) * nmat.ptr()[i];
	}
	scale = fact * _smoothStartScale + (1.0 - fact) * nscale;
	return true;
    }*/
    //else
    {
	return mRecordEntryListPtr->lookupRecordEntry(mTimer, scale, xMat);
    }
}

























