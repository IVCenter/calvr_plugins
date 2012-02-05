/***************************************************************
* File Name: AudioConfigHandler.h
*
* Class Name: AudioConfigHandler
*
***************************************************************/

#ifndef _AUDIO_CONFIG_HANDLER_H_
#define _AUDIO_CONFIG_HANDLER_H_


// C++
#include <stdarg.h>
#include <stdlib.h>
#include <iostream>
#include <list>
#include <netdb.h>
#include <sys/socket.h>

// CalVR plugin support
#include <config/ConfigManager.h>

// Open scene graph
#include <osg/Switch>

// local includes
#include "Audio/OSCPack.h"
#include "Geometry/CAVEGeodeShape.h"
#include "Geometry/CAVEGroupShape.h"


/***************************************************************
* Class: AudioConfigHandler
***************************************************************/
class AudioConfigHandler
{
  public:
    AudioConfigHandler(osg::Switch *shapeSwitchPtr);
    ~AudioConfigHandler();

    /* communications between covise and audio server: functions only called when 'mFlagMaster' is set. */
    void connectServer();
    void disconnectServer();
    bool isConnected() { return mFlagConnected; }
    void setMasterFlag(bool flag) { mFlagMaster = flag; }

    /* update functions */
    void updateShapes();
    void updatePoses(const osg::Vec3 &viewDir, const osg::Vec3 &viewPos);

  protected:
    bool mFlagConnected, mFlagMaster;
    int mSockfd;

    /* Swich pointer passed from 'DesignObjectHandler' that holds all registared CAVE geometries */
    osg::Switch *mCAVEShapeSwitch;

};


#endif

