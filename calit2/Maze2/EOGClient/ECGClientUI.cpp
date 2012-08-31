/***************************************************************
* File Name: ECGClientUI.cpp
*
* Description: 
*
* Written by ZHANG Lelin on July 21, 2010
*
***************************************************************/
#include "ECGClient.h"


using namespace std;
using namespace cvr;


/***************************************************************
*  Function: initVRUIMenu()
*
*  'mainMenuItem' is added as a sub menu in CaveCAD plugin
*
***************************************************************/
void ECGClient::initCVRMenu(SubMenu *mainMenuMaze2)
{
    /* create main menu handler */    
    mainMenu = new SubMenu("ECG Data Client", "ECG Data Client");
    mainMenuMaze2->addItem(mainMenu);

    /* Main row menu items: Press buttons and checkboxes */
    connectToServerCheckbox = new MenuCheckbox("Connect to Data Server", false);
    connectToServerCheckbox->setCallback(this);
    mainMenu->addItem(connectToServerCheckbox);

    connectionStatusLabel = new MenuText("Connection Status");
    connectionStatusLabel->setText("   Disconnected from Data Server.");
    mainMenu->addItem(connectionStatusLabel);

    startStopCalibrationCheckbox = new MenuCheckbox("Start/Stop Calibration", false);
    startStopCalibrationCheckbox->setCallback(this);
    mainMenu->addItem(startStopCalibrationCheckbox);

	startStopPlaybackCheckbox = new MenuCheckbox("Start/Stop Playback", false);
	startStopPlaybackCheckbox->setCallback(this);
	mainMenu->addItem(startStopPlaybackCheckbox);

	playbackTimerLabel = new MenuText("Time = 00.000 s");
	mainMenu->addItem(playbackTimerLabel);

    showCalibrationBallCheckbox = new MenuCheckbox("Show Calibration Ball", false);
    showCalibrationBallCheckbox->setCallback(this);
    mainMenu->addItem(showCalibrationBallCheckbox);

	showPlaybackBallCheckbox = new MenuCheckbox("Show Playback Ball", false);
    showPlaybackBallCheckbox->setCallback(this);
    mainMenu->addItem(showPlaybackBallCheckbox);

    showCalibrationFieldCheckbox = new MenuCheckbox("Show Calibration Field", false);
    showCalibrationFieldCheckbox->setCallback(this);
    mainMenu->addItem(showCalibrationFieldCheckbox);

    resetBallButtonMenuItem = new MenuButton("Reset Ball Position");
    resetBallButtonMenuItem->setCallback(this);
    mainMenu->addItem(resetBallButtonMenuItem);

    alignWithViewerButtonMenuItem = new MenuButton("Align with Viewer's Position");
    alignWithViewerButtonMenuItem->setCallback(this);
    mainMenu->addItem(alignWithViewerButtonMenuItem);

    /* Main row menu items: Slider bars */
    leftRangeSlider = new MenuRangeValue("Left Side Range", 0, 20, 0.5);
    leftRangeSlider->setCallback(this);
    leftRangeSlider->setValue(20);
    mainMenu->addItem(leftRangeSlider);

    rightRangeSlider = new MenuRangeValue("Right Side Range", 0, 20, 0.5);
    rightRangeSlider->setCallback(this);
    rightRangeSlider->setValue(20);
    mainMenu->addItem(rightRangeSlider);

    upwardRangeSlider = new MenuRangeValue("Upward Range", 0, 20, 0.5);
    upwardRangeSlider->setCallback(this);
    upwardRangeSlider->setValue(20);
    mainMenu->addItem(upwardRangeSlider);

    downwardRangeSlider = new MenuRangeValue("Downward Range", 0, 20, 0.5);
    downwardRangeSlider->setCallback(this);
    downwardRangeSlider->setValue(20);
    mainMenu->addItem(downwardRangeSlider);

    depthRangeSlider = new MenuRangeValue("Depth Range", 0.5, 5, 0.5);
    depthRangeSlider->setCallback(this);
    depthRangeSlider->setValue(5);
    mainMenu->addItem(depthRangeSlider);

    horizontalFreqSlider = new MenuRangeValue("Horizontal Frequency", 0, 0.2, 0.1);//1.0, 0.1);
    horizontalFreqSlider->setCallback(this);
    mainMenu->addItem(horizontalFreqSlider);

    verticalFreqSlider = new MenuRangeValue("Vertical Frequency", 0, 0.2, 0.1);//1.0, 0.1);
    verticalFreqSlider->setCallback(this);
    mainMenu->addItem(verticalFreqSlider);

    depthFreqSlider = new MenuRangeValue("Depth Frequency", 0, 0.2, 0.1);//1.0, 0.1);
    depthFreqSlider->setCallback(this);
    mainMenu->addItem(depthFreqSlider);
}


/***************************************************************
*  Function: menuCallback()
***************************************************************/
void ECGClient::menuCallback(MenuItem *item)
{
    /* connect / disconnect from data server */
    if (item == connectToServerCheckbox)
    {
      	if (connectToServerCheckbox->getValue())
      	{
	    	connectionStatusLabel->setText("   Connecting to Data Server.");
	    	if(mFlagMaster) connectServer();
	    	connectionStatusLabel->setText("   Connected to Data Server.");
      	} else {
	    	connectionStatusLabel->setText("   Disconnected from Data Server.");
	    	if(mFlagMaster) disconnectServer();
		}
    } 

    /* start / stop calibration ball movement */
    else if (item == startStopCalibrationCheckbox)
    {
      	if (startStopCalibrationCheckbox->getValue())
      	{
			if (startStopPlaybackCheckbox->getValue())
			{
				startStopPlaybackCheckbox->setValue(false);
				mCaliController->stopPlayback();
			}
	    	mCaliController->startCalibration();
      	} else {
	    	mCaliController->stopCalibration();
		}
    }

	/* start / stop data playback */
	else if (item == startStopPlaybackCheckbox)
	{
		if (startStopPlaybackCheckbox->getValue())
      	{
			if (startStopCalibrationCheckbox->getValue())
			{
				startStopCalibrationCheckbox->setValue(false);
				mCaliController->stopCalibration();
			}
	    	mCaliController->startPlayback();
      	} else {
	    	mCaliController->stopPlayback();
		}
	}

    /* show / hide calibration ball */
    else if (item == showCalibrationBallCheckbox)
    {
      	if (showCalibrationBallCheckbox->getValue())
      	{
	    	mCaliController->setCaliBallVisible(true);
	    	mCaliController->resetCaliField(mInvBaseMat);
      	} else {
	    	mCaliController->setCaliBallVisible(false);
		}
    }

	/* show / hide playback ball */
    else if (item == showPlaybackBallCheckbox)
    {
      	if (showPlaybackBallCheckbox->getValue())
      	{
	    	mCaliController->setPlaybackBallVisible(true);
      	} else {
	    	mCaliController->setPlaybackBallVisible(false);
		}
    }

    /* show / hide calibration filed */
    else if (item == showCalibrationFieldCheckbox)
    {
      	if (showCalibrationFieldCheckbox->getValue())
      	{
	    	mCaliController->setCaliFieldVisible(true);
      	} else {
	    	mCaliController->setCaliFieldVisible(false);
		}
    }

    /* reset ball to its default position */
    else if (item == resetBallButtonMenuItem)
    {
		mCaliController->resetCaliBall();
    }

    /* align calibration field with viewer's front position */
    else if (item == alignWithViewerButtonMenuItem)
    {
		mCaliController->resetCaliField(mInvBaseMat);
    }

    /* set ranges and frequencies of calibration trajectory */
    else if (item == leftRangeSlider)
    {
		float val = leftRangeSlider->getValue();
		mCaliController->updateCaliParam(CalibrationController::LEFT_RANGE, val);
    }
    else if (item == rightRangeSlider)
    {
		float val = rightRangeSlider->getValue();
		mCaliController->updateCaliParam(CalibrationController::RIGHT_RANGE, val);
    }
    else if (item == upwardRangeSlider)
    {
		float val = upwardRangeSlider->getValue();
		mCaliController->updateCaliParam(CalibrationController::UPWARD_RANGE, val);
    }
    else if (item == downwardRangeSlider)
    {
		float val = downwardRangeSlider->getValue();
		mCaliController->updateCaliParam(CalibrationController::DOWNWARD_RANGE, val);
    }
    else if (item == depthRangeSlider)
    {
		float val = depthRangeSlider->getValue();
		mCaliController->updateCaliParam(CalibrationController::DEPTH_RANGE, val);
    }

    else if (item == horizontalFreqSlider)
    {
		float val = horizontalFreqSlider->getValue();
		mCaliController->updateCaliParam(CalibrationController::HORIZONTAL_FREQ, val);
    }
    else if (item == verticalFreqSlider)
    {
		float val = verticalFreqSlider->getValue();
		mCaliController->updateCaliParam(CalibrationController::VERTICAL_FREQ, val);
    }
    else if (item == depthFreqSlider)
    {
		float val = depthFreqSlider->getValue();
		mCaliController->updateCaliParam(CalibrationController::DEPTH_FREQ, val);
    }
}


















