/***************************************************************
* File Name: PathRecorder.cpp
*
* Description: Record/playback user-guided navigation paths
*
* Written by ZHANG Lelin on Oct 15, 2010
*
***************************************************************/
#include "PathRecorder.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>

CVRPLUGIN(PathRecorder)

using namespace osg;
using namespace std;

PathRecorder *plugin = NULL;

//Constructor
PathRecorder::PathRecorder()
{
}

//Destructor
PathRecorder::~PathRecorder()
{
}


/***************************************************************
*  Function: init()
***************************************************************/
bool PathRecorder::init()
{
    mDataDir = ConfigManager::getEntry("Plugin.PathRecorder.DataDir");

    plugin = this;
    mPathRecordManager = new PathRecordManager(mDataDir);
    initUI();
    return true;
}


/***************************************************************
*  Function: preFrame()
***************************************************************/
void PathRecorder::preFrame()
{
    /* update timer of mPathRecordManager */
    double frameDuration = PluginHelper::getLastFrameDuration();
    mPathRecordManager->updateTimer(frameDuration);

    /* read / write X matrix and scale values */
    if (mPathRecordManager->getState() == PathRecordManager::RECORD)
    {
	if (mPathRecordManager->isPlayRecord())
	{
	    double scale = PluginHelper::getObjectScale();
	    Matrixd xMat = PluginHelper::getObjectMatrix();
	    mPathRecordManager->recordPathEntry(scale, xMat);
	}
    }
    else if (mPathRecordManager->getState() == PathRecordManager::PLAYBACK)
    {
	if (mPathRecordManager->isPlayRecord())
	{
	    double scale;
	    Matrixd xMat;
	    if (mPathRecordManager->playbackPathEntry(scale, xMat))
	    {
		PluginHelper::setObjectScale(scale);
		PluginHelper::setObjectMatrix(xMat);
	    }
	}
    }
}


















