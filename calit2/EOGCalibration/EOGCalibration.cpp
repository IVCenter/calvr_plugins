/***************************************************************
* File Name: EOGCalibration.cpp
*
***************************************************************/
#include "EOGCalibration.h"


using namespace std;
using namespace cvr;

CVRPLUGIN(EOGCalibration)

/***************************************************************
*  Constructor: EOGCalibration()
***************************************************************/
EOGCalibration::EOGCalibration(): mFlagConnected(false), mFlagMaster(false)
{
    string DataDir = ConfigManager::getEntry("Plugin.EOGCalibration.DataDir");
    
    _rootGroup = new osg::Group();
    PluginHelper::getObjectsRoot()->addChild(_rootGroup);

    mCaliController = new CalibrationController(_rootGroup, DataDir);
    mCaliController->stopCalibration();

    if (cvr::ComController::instance()->isMaster())
        mFlagMaster = true;
}

/***************************************************************
*  Destrutor: ~EOGCalibration()
***************************************************************/
EOGCalibration::~EOGCalibration()
{
    if (mSockfd) 
        close(mSockfd);
}

/***************************************************************
*  Function: init()
***************************************************************/
bool EOGCalibration::init()
{
	mainMenu = new SubMenu("EOGCalibration", "EOGCalibration");
	MenuSystem::instance()->addMenuItem(mainMenu);

    connectToServerCheckbox = new MenuCheckbox("Connect to Data Server", false);
    connectToServerCheckbox->setCallback(this);
    mainMenu->addItem(connectToServerCheckbox);

    connectionStatusLabel = new MenuText("Connection Status");
    connectionStatusLabel->setText("Disconnected from Data Server.");
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
void EOGCalibration::menuCallback(cvr::MenuItem *item)
{
    // connect / disconnect from data server
    if (item == connectToServerCheckbox)
    {
      	if (connectToServerCheckbox->getValue())
      	{
	    	connectionStatusLabel->setText("   Connecting to Data Server.");
	    	if(mFlagMaster) 
                connectServer();
	    	connectionStatusLabel->setText("       Connected to Data Server.");
      	} 
        else 
        {
	    	connectionStatusLabel->setText("Disconnected from Data Server.");
	    	if(mFlagMaster) disconnectServer();
		}
    } 

    // start / stop calibration ball movement
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
      	} 
        else 
        {
	    	mCaliController->stopCalibration();
		}
    }

	// start / stop data playback
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
      	} 
        else 
        {
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
      	} 
        else 
        {
	    	mCaliController->setCaliBallVisible(false);
		}
    }

	/* show / hide playback ball */
    else if (item == showPlaybackBallCheckbox)
    {
      	if (showPlaybackBallCheckbox->getValue())
      	{
	    	mCaliController->setPlaybackBallVisible(true);
      	} 
        else 
        {
	    	mCaliController->setPlaybackBallVisible(false);
		}
    }

    /* show / hide calibration filed */
    else if (item == showCalibrationFieldCheckbox)
    {
      	if (showCalibrationFieldCheckbox->getValue())
      	{
	    	mCaliController->setCaliFieldVisible(true);
      	} 
        else 
        {
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

/***************************************************************
*  Function: preFrame()
***************************************************************/
void EOGCalibration::preFrame()
{
    Matrixf invBaseMat = PluginHelper::getWorldToObjectTransform();
    Matrixf viewMat = PluginHelper::getHeadMat(0);
    Matrixf xformMat = PluginHelper::getObjectMatrix();
 
    /* get viewer's position in world space */
    Vec3 viewOrg = viewMat.getTrans() * invBaseMat; 
    Vec3 viewPos = Vec3(0.0, 1.0, 0.0) * viewMat * invBaseMat; 
    Vec3 viewDir = viewPos - viewOrg;
    viewDir.normalize();

    //mAudioConfigHandler->updatePoses(viewDir, viewPos);
    //mNaviHandler->updateNaviStates(Navigation::instance()->getScale(), viewDir, viewPos);
    //mNaviHandler->updateButtonStates();
    //mNaviHandler->updateXformMat();

    /* ECGClient: Master-Slave operations: Read current time for update */
    double frameDuration;
    if(ComController::instance()->isMaster())
    {
		frameDuration = PluginHelper::getLastFrameDuration();
		((double*)mClkBuf)[0] = frameDuration;
		cvr::ComController::instance()->sendSlaves((char*) &mClkBuf, sizeof(mClkBuf));
    } 
    else 
    {
		cvr::ComController::instance()->readMaster((char*) &mClkBuf, sizeof(mClkBuf));
		frameDuration = ((double*)mClkBuf)[0];
    }

    /* update inverse base matrix for re-aligning calibration field */
    mInvBaseMat = viewMat * invBaseMat;
    mCaliController->updateViewMat(viewMat);

    /*  DEBUGGING OUTPUT: Head position and orientations in CAVE space
    Vec3 rightVec = Vec3(viewMat(0, 0), viewMat(0, 1), viewMat(0, 2));
    Vec3 frontVec= Vec3(viewMat(1, 0), viewMat(1, 1), viewMat(1, 2));
    Vec3 upVec = Vec3(viewMat(2, 0), viewMat(2, 1), viewMat(2, 2));
    Vec3 headPos = Vec3(viewMat(3, 0), viewMat(3, 1), viewMat(3, 2)) / 1000.f;
    cerr << "Right vector = " << rightVec.x() << " " << rightVec.y() << " " << rightVec.z() << endl;
    cerr << "Front vector = " << frontVec.x() << " " << frontVec.y() << " " << frontVec.z() << endl;
    cerr << "Up vector = " << upVec.x() << " " << upVec.y() << " " << upVec.z() << endl;  
    cerr << "Head position = " << headPos.x() << " " << headPos.y() << " " << headPos.z() << endl;
    // cerr << "phi = " << phi << "    theta = " << theta << "    rad = " << rad << endl;
    cerr << endl; 

    DEBUGGING OUTPUT: Viewer positions and orientations in world space
    Vec3 rightVec = Vec3(xformMat(0, 0), xformMat(0, 1), xformMat(0, 2));
    Vec3 frontVec= Vec3(xformMat(1, 0), xformMat(1, 1), xformMat(1, 2));
    Vec3 upVec = Vec3(xformMat(2, 0), xformMat(2, 1), xformMat(2, 2));
    Vec3 headPos = Vec3(xformMat(0, 3), xformMat(1, 3), xformMat(2, 3)) / 1000.f * (-1);
    cerr << "Right vector = " << rightVec.x() << " " << rightVec.y() << " " << rightVec.z() << endl;
    cerr << "Front vector = " << frontVec.x() << " " << frontVec.y() << " " << frontVec.z() << endl;
    cerr << "Up vector = " << upVec.x() << " " << upVec.y() << " " << upVec.z() << endl;  
    cerr << "Head position = " << headPos.x() << " " << headPos.y() << " " << headPos.z() << endl;
    cerr << endl; */

	float phi, theta, rad;
    if (mCaliController->isCaliBallVisible())
	{
    	mCaliController->updateCaliTime(frameDuration);
    	mCaliController->updateCaliBallPos(phi, theta, rad);
	}
	if (mCaliController->isPlaybackBallVisible())
	{
		mCaliController->updatePlaybackTime(frameDuration);
		mCaliController->updatePlaybackBallPos();
		playbackTimerLabel->setText(mCaliController->getPlaybackTimeLabel());
	}

    /* create client message and send to data server */
    if (mFlagConnected && mFlagMaster && mCaliController->isCalibrationStarted() && mCaliController->isCaliBallVisible())
    {
        Vec3 headPos = Vec3(viewMat(3, 0), viewMat(3, 1), viewMat(3, 2)) / 1000.f;
        struct MSGClient msg(viewMat(0, 0), viewMat(0, 1), viewMat(0, 2), viewMat(1, 0), viewMat(1, 1), viewMat(1, 2),
                     viewMat(2, 0), viewMat(2, 1), viewMat(2, 2), headPos.x(), headPos.y(), headPos.z(),
                     phi, theta, rad);

        int nBytes = send(mSockfd, &msg, sizeof (MSGClient), 0);
        if (nBytes != sizeof (MSGClient)) 
        {
            std::cerr << "EOGCalibration Warning: Lost data transmission. nBytes = " << nBytes << std::endl;
        }
    }

}

/***************************************************************
*  Function: processEvent()
***************************************************************/
bool EOGCalibration::processEvent(cvr::InteractionEvent *event)
{
    return false;
}

/***************************************************************
*  Function: connectServer()
***************************************************************/
void EOGCalibration::connectServer()
{
    /* get server address and port number */
    string server_addr = ConfigManager::getEntry("Plugin.EOGCalibration.EOGDataServerAddress");
    int port_number = ConfigManager::getInt("Plugin.EOGCalibration.EOGDataServerPort", 8084);
    if(server_addr == "") 
        server_addr = "127.0.0.1";
    
    std::cerr << "EOGCalibration::Server address: " << server_addr << std::endl;
    std::cerr << "EOGCalibration::Port number: " << port_number << std::endl;

    /* build up socket communications */
    int portno = port_number, protocol = SOCK_STREAM;
    struct sockaddr_in server;
    struct hostent *hp;

    mSockfd = socket(AF_INET, protocol, 0);
    if (mSockfd < 0)
    {
        std::cerr << "EOGCalibration::connect(): Can't open socket." << std::endl;
        return;
    }

    server.sin_family = AF_INET;
    hp = gethostbyname(server_addr.c_str());
    if (hp == 0)
    {
        std::cerr << "EOGCalibration::connect(): Unknown host." << std::endl;
        close(mSockfd);
        return;
    }
    memcpy((char *)&server.sin_addr, (char *)hp->h_addr, hp->h_length);
    server.sin_port = htons((unsigned short)portno);

    /* connect to ECG data server */
    if (connect(mSockfd, (struct sockaddr *) &server, sizeof (server)) < 0)
    {
        std::cerr << "EOGCalibration::connect(): Failed connect to server" << std::endl;
        close(mSockfd);
        return;
    }

    std::cerr << "EOGCalibration::Successfully connected to EOG Data Server." << std::endl;
    mFlagConnected = true;
}

/***************************************************************
*  Function: disconnectServer()
***************************************************************/
void EOGCalibration::disconnectServer()
{
    close(mSockfd);
    mFlagConnected = false;
    mSockfd = 0;
}

