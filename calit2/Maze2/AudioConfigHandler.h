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
#include <unistd.h>

// open scene graph
#include <osg/Vec3>
#include <osg/Matrixd>

// CalVR plugin support
#include <cvrConfig/ConfigManager.h>

// local includes
#include "Audio/OSCPack.h"
#include "AcoustiMaze/AcoustiMaze.h"


/***************************************************************
* Class: AudioConfigHandler
***************************************************************/
class AudioConfigHandler
{
  public:
    AudioConfigHandler();
    ~AudioConfigHandler();

    /* communications between calvr and audio server: functions only called when 'mFlagMaster' is set. */
    void connectServer();
    void disconnectServer();
    bool isConnected() { return mFlagConnected; }
    void setMasterFlag(bool flag) { mFlagMaster = flag; }

    /* update functions */
    void loadSoundSource();
    void updateGeometry();
    void updatePoses(const osg::Vec3 &viewDir, const osg::Vec3 &viewPos);

  protected:
    bool mFlagConnected, mFlagMaster;
    int mSockfd;
    AcoustiMaze * _acoustiMaze;
};


#endif
