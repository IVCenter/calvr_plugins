/***************************************************************
* File Name: EOGCalibration.h
*
***************************************************************/
#ifndef EOG_CALIBRATION_H
#define EOG_CALIBRATION_H

// C++
#include <netdb.h>
#include <sys/socket.h>

// Open Scene Graph
#include <osg/Matrixf>

// CalVR plugin support
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/ComController.h>

// CalVR menu system
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuText.h>

// Local:
#include "CalibrationController.h"


using namespace std;
using namespace osg;

/***************************************************************
*  Class: EOGCalibration
***************************************************************/
class EOGCalibration : public cvr::CVRPlugin, public cvr::MenuCallback
{
  public:
    EOGCalibration();
    virtual ~EOGCalibration();
    static EOGCalibration * instance();

    bool init();
	virtual void preFrame();

	void menuCallback(cvr::MenuItem *item);
	bool processEvent(cvr::InteractionEvent *event);

    void connectServer();
    void disconnectServer();
    bool isConnected() { return mFlagConnected; }
    void setMasterFlag(bool flag) { mFlagMaster = flag; }
    
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
    cvr::SubMenu *mainMenu, *optionsMenu;
    cvr::MenuCheckbox 	*connectToServerCheckbox, *startStopCalibrationCheckbox, *startStopPlaybackCheckbox,
						*showCalibrationBallCheckbox, *showPlaybackBallCheckbox, *showCalibrationFieldCheckbox,
                        *appearTestCheckbox;
    cvr::MenuText *connectionStatusLabel, *playbackTimerLabel;
    cvr::MenuButton *resetBallButtonMenuItem, *alignWithViewerButtonMenuItem;
    cvr::MenuRangeValue *leftRangeSlider, *rightRangeSlider, *upwardRangeSlider, *downwardRangeSlider, *depthRangeSlider, *horizontalFreqSlider, *verticalFreqSlider, *depthFreqSlider;

  protected:
    CalibrationController *mCaliController;
    Matrixf mInvBaseMat;
    osg::Group * _rootGroup;
    static EOGCalibration * _myPtr;
    bool mFlagConnected, mFlagMaster;
    int mSockfd;
    double mClkBuf[2];
};

#endif

