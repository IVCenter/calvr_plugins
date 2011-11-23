/***************************************************************
* File Name: ECGClient.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Jun 29, 2010
*
***************************************************************/
#include "ECGClient.h"


using namespace std;
using namespace cvr;


/***************************************************************
*  Constructor: ECGClient()
***************************************************************/
ECGClient::ECGClient(Group *rootGroup): mFlagConnected(false), mFlagMaster(false)
{
    string DataDir = ConfigManager::getEntry("Plugin.Maze2.DataDir");

    mCaliController = new CalibrationController(rootGroup, DataDir);
    mCaliController->stopCalibration();
}


/***************************************************************
*  Destrutor: ~ECGClient()
***************************************************************/
ECGClient::~ECGClient()
{
    if (mSockfd) close(mSockfd);
}


/***************************************************************
*  Function: connectServer()
***************************************************************/
void ECGClient::connectServer()
{
    /* get server address and port number */
    string server_addr = ConfigManager::getEntry("Plugin.Maze2.EOGDataServerAddress");
    int port_number = ConfigManager::getInt("Plugin.Maze2.EOGDataServerPort", 8084);
    if( server_addr == "" ) server_addr = "127.0.0.1";
    
    cerr << "Maze2::ECGClient::Server address: " << server_addr << endl;
    cerr << "Maze2::ECGClient::Port number: " << port_number << endl;

    /* build up socket communications */
    int portno = port_number, protocol = SOCK_STREAM;
    struct sockaddr_in server;
    struct hostent *hp;

    mSockfd = socket(AF_INET, protocol, 0);
    if (mSockfd < 0)
    {
        cerr << "Maze2::ECGClient::connect(): Can't open socket." << endl;
        return;
    }

    server.sin_family = AF_INET;
    hp = gethostbyname(server_addr.c_str());
    if (hp == 0)
    {
        cerr << "Maze2::ECGClient::connect(): Unknown host." << endl;
        close(mSockfd);
        return;
    }
    memcpy((char *)&server.sin_addr, (char *)hp->h_addr, hp->h_length);
    server.sin_port = htons((unsigned short)portno);

    /* connect to ECG data server */
    if (connect(mSockfd, (struct sockaddr *) &server, sizeof (server)) < 0)
    {
	cerr << "Maze2::ECGClient::connect(): Failed connect to server" << endl;
        close(mSockfd);
        return;
    }

    cerr << "Maze2::ECGClient::Successfully connected to EOG Data Server." << endl;
    mFlagConnected = true;
}


/***************************************************************
*  Function: disconnectServer()
***************************************************************/
void ECGClient::disconnectServer()
{
    close(mSockfd);
    mFlagConnected = false;
    mSockfd = 0;
}


/***************************************************************
*  Function: update()
*
*  1) Extract viewing vectors from view matrix
*  2) Update calibration ball position, get phase parameters
*  3) Send data to ECG data server (Master only)
*
***************************************************************/
void ECGClient::update(const Matrixf &viewMat, const Matrixf &invBaseMat, const Matrixf &xformMat, const double &frameDuration)
{
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
    cerr << endl; */

    /*  DEBUGGING OUTPUT: Viewer positions and orientations in world space
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
	if (nBytes != sizeof (MSGClient)) cerr  << "ECGClient Warning: Lost data transmission. nBytes = " 
						<< nBytes << endl;
    }
}










