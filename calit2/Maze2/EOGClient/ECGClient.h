/***************************************************************
* File Name: ECGClient.h
*
* Class Name: ECGClient
*
***************************************************************/
#ifndef _ECG_CLIENT_H_
#define _ECG_CLIENT_H_

// C++
#include <netdb.h>
#include <sys/socket.h>

// Open Scene Graph
#include <osg/Matrixf>

// CalVR plugin support
#include <config/ConfigManager.h>

// CalVR menu system
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuRangeValue.h>
#include <menu/MenuText.h>

// Local:
#include "CalibrationController.h"


using namespace std;
using namespace osg;


/***************************************************************
*  Class: ECGClient
***************************************************************/
class ECGClient: public cvr::MenuCallback
{
  public:
    ECGClient(Group *rootGroup);
    ~ECGClient();

    void connectServer();
    void disconnectServer();
    bool isConnected() { return mFlagConnected; }
    void setMasterFlag(bool flag) { mFlagMaster = flag; }
    void update(const Matrixf &viewMat, const Matrixf &invBaseMat, const Matrixf &xformMat, const double &frameDuration);
    
    /* Message data structure from client */
    struct MSGClient
    {
		MSGClient(float rx, float ry, float rz, float fx, float fy, float fz,
		  		float ux, float uy, float uz, float hx, float hy, float hz,
		  		float ph, float th, float ra)
		{
	    	rightX = rx;  rightY = ry;  rightZ = rz;
	    	frontX = fx;  frontY = fy;  frontZ = fz;
	    	upX = ux; upY = uy; upZ = uz;
	    	headX = hx;  headY = hy;  headZ = hz;
	    	phi = ph;  theta = th;  rad = ra;
		}

		float rightX, rightY, rightZ;
		float frontX, frontY, frontZ;
		float upX, upY, upZ;
		float headX, headY, headZ;
		float phi, theta, rad;
    };

    /* CalVR Menu Items */
    void initCVRMenu(cvr::SubMenu *mainMenuMaze2);
	virtual void menuCallback(cvr::MenuItem *item);

    cvr::SubMenu *mainMenu;
    cvr::MenuCheckbox 	*connectToServerCheckbox, *startStopCalibrationCheckbox, *startStopPlaybackCheckbox,
						*showCalibrationBallCheckbox, *showPlaybackBallCheckbox, *showCalibrationFieldCheckbox;
    cvr::MenuText *connectionStatusLabel, *playbackTimerLabel;
    cvr::MenuButton *resetBallButtonMenuItem, *alignWithViewerButtonMenuItem;
    cvr::MenuRangeValue *leftRangeSlider, *rightRangeSlider, *upwardRangeSlider, *downwardRangeSlider, *depthRangeSlider, *horizontalFreqSlider, *verticalFreqSlider, *depthFreqSlider;

  protected:
    CalibrationController *mCaliController;
    Matrixf mInvBaseMat;

    bool mFlagConnected, mFlagMaster;
    int mSockfd;
};

#endif



