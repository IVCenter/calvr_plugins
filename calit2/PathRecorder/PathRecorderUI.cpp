/***************************************************************
* File Name: PathRecorderUI.cpp
*
* Description: Initialize viewport for Jacobs Hospital 
*
* Written by ZHANG Lelin on Oct 15, 2010
*
***************************************************************/

#include "PathRecorder.h"

#include <kernel/PluginHelper.h>
#include <kernel/PluginManager.h>

#include <PluginMessageType.h>

using namespace osg;
using namespace std;


/***************************************************************
* Function: initCOVERUI
***************************************************************/
void PathRecorder::initUI()
{
    mainMenu = new SubMenu("Path Recorder","Path Recorder");
    PluginHelper::addRootMenuItem(mainMenu);

    selectPathFilesMenu = new SubMenu("Select Path Files","Select Path Files");
    mainMenu->addItem(selectPathFilesMenu);

    int numFiles = mPathRecordManager->getNumFiles();
    recordEntryCheckboxList = new MenuCheckbox*[numFiles];

    for (int i = 0; i < numFiles; i++)
    {
	recordEntryCheckboxList[i] = new MenuCheckbox(mPathRecordManager->getFilename(i), false);
	recordEntryCheckboxList[i]->setCallback(this);
	selectPathFilesMenu->addItem(recordEntryCheckboxList[i]);
    }
    recordEntryCheckboxList[0]->setValue(true);
    mPathRecordManager->setActiveFileIdx(0);

    /* main menu items */
    recordPathCheckbox = new MenuCheckbox("Opt 1: Record path", true);
    recordPathCheckbox->setCallback(this);
    mainMenu->addItem(recordPathCheckbox);

    playbackPathCheckbox = new MenuCheckbox("Opt 2: Playback path", false);
    playbackPathCheckbox->setCallback(this);
    mainMenu->addItem(playbackPathCheckbox);

    mainMenu->addItem(mPathRecordManager->getInfoLabelPtr());

    playbackSpeedSlider = new MenuRangeValue("Playback speed", 0.0, 4.0, 0);
    playbackSpeedSlider->setCallback(this);
    playbackSpeedSlider->setValue(1.0);
    mainMenu->addItem(playbackSpeedSlider);

    startButton = new MenuButton("Start");
    startButton->setCallback(this);
    mainMenu->addItem(startButton);

    pauseButton = new MenuButton("Pause");
    pauseButton->setCallback(this);
    mainMenu->addItem(pauseButton);

    stopButton = new MenuButton("Stop");
    stopButton->setCallback(this);
    mainMenu->addItem(stopButton);
}


/***************************************************************
*  Function: menuEvent()
***************************************************************/
void PathRecorder::menuCallback(MenuItem *activeItem)
{
    int numFiles = mPathRecordManager->getNumFiles(); 
    for (int i = 0; i < numFiles; i++)
    {
	if (activeItem == recordEntryCheckboxList[i])
	{
	    if (recordEntryCheckboxList[i]->getValue())
	    {
		int activeIdx = mPathRecordManager->getActiveFileIdx();
		if (activeIdx >= 0) recordEntryCheckboxList[activeIdx]->setValue(false);
		mPathRecordManager->setActiveFileIdx(i);
	    } else mPathRecordManager->setActiveFileIdx(-1);
	}
    }

    // set to option 1: record path
    if (activeItem == recordPathCheckbox)
    {
      	if (recordPathCheckbox->getValue())
      	{
	    mPathRecordManager->setState(PathRecordManager::RECORD);
	    playbackPathCheckbox->setValue(false);
      	} else {
	    mPathRecordManager->setState(PathRecordManager::SPARE);
	}
    } 
    // set to option 2: playback path
    if (activeItem == playbackPathCheckbox)
    {
      	if (playbackPathCheckbox->getValue())
      	{
	    mPathRecordManager->setState(PathRecordManager::PLAYBACK);
	    recordPathCheckbox->setValue(false);
      	} else {
	    mPathRecordManager->setState(PathRecordManager::SPARE);
	}
    }

    // set playback speed
    if (activeItem == playbackSpeedSlider)
    {
	double speed = playbackSpeedSlider->getValue();
	mPathRecordManager->setPlaybackSpeed(speed);
    }

    // start recording/playback
    if (activeItem == startButton)
    {
	mPathRecordManager->start();
    }
    // pause recording/playback
    if (activeItem == pauseButton)
    {
	mPathRecordManager->pause();
    }
    // stop recording/playback
    if (activeItem == stopButton)
    {
	mPathRecordManager->stop();
    }
}

void PathRecorder::message(int type, char * data)
{
    PathRecorderMessageType mtype = (PathRecorderMessageType)type;
    switch(mtype)
    {
	case PR_SET_RECORD:
	{
	    bool val = *((bool*)data);
	    if(val == recordPathCheckbox->getValue())
	    {
		break;
	    }

	    if(val)
	    {
		mPathRecordManager->setState(PathRecordManager::RECORD);
		playbackPathCheckbox->setValue(false);
	    }
	    else
	    {
		mPathRecordManager->setState(PathRecordManager::SPARE);
	    }
	    recordPathCheckbox->setValue(val);
	    break;
	}
	case PR_SET_PLAYBACK:
	{
	    bool val = *((bool*)data);
	    if(val == playbackPathCheckbox->getValue())
	    {
		break;
	    }

	    if(val)
	    {
		mPathRecordManager->setState(PathRecordManager::PLAYBACK);
		recordPathCheckbox->setValue(false);
	    }
	    else
	    {
		mPathRecordManager->setState(PathRecordManager::SPARE);
	    }
	    playbackPathCheckbox->setValue(val);
	    break;
	}
	case PR_SET_ACTIVE_ID:
	{
	    int index = *((int*)data);
	    int activeIdx = mPathRecordManager->getActiveFileIdx();
	    if (activeIdx >= 0) recordEntryCheckboxList[activeIdx]->setValue(false);
	    mPathRecordManager->setActiveFileIdx(index);
	    if(index >= 0)
	    {
		recordEntryCheckboxList[index]->setValue(true);
	    }
	    break;
	}
	case PR_SET_PLAYBACK_SPEED:
	{
	    double speed = *((float *)data);
            std::cerr << "Playback setting speed to: " << speed << std::endl;
	    mPathRecordManager->setPlaybackSpeed(speed);
	    break;
	}
	case PR_START:
	{
	    mPathRecordManager->start();
	    break;
	}
	case PR_PAUSE:
	{
	    mPathRecordManager->pause();
	    break;
	}
	case PR_STOP:
	{
	    mPathRecordManager->stop();
	    break;
	}
	case PR_GET_TIME:
	{
	    double time = mPathRecordManager->getTime();
	    PluginManager::instance()->sendMessageByName(data,PR_GET_TIME,(char*)&time);
	    break;
	}
	case PR_GET_START_MAT:
	{
	    if(mPathRecordManager->getActiveFileIdx() >= 0)
	    {
		osg::Matrixd m;
		double f;
		mPathRecordManager->playbackPathEntry(f, m);

		PluginManager::instance()->sendMessageByName(data,PR_GET_START_MAT,(char*)m.ptr());
	    }
	    break;
	}
	case PR_GET_START_SCALE:
	{
	    if(mPathRecordManager->getActiveFileIdx() >= 0)
	    {
		osg::Matrix m;
		double f;
		mPathRecordManager->playbackPathEntry(f, m);

                float f1 = f;

		PluginManager::instance()->sendMessageByName(data,PR_GET_START_SCALE,(char*)&f1);
	    }
	    break;
	}
        case PR_IS_STOPPED:
        {
            bool b = false;
            if(mPathRecordManager->getPlaybackState() == PathRecordManager::DONE)
            {
                b = true;
            }
            PluginManager::instance()->sendMessageByName(data,PR_IS_STOPPED,(char*)&b);
            break;
        }
	default:
	    break;
    }
}











