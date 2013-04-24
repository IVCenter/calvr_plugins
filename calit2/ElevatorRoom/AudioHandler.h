#ifndef AUDIO_HANDLER_H
#define AUDIO_HANDLER_H


// C++
#include <stdlib.h>
#include <iostream>
#include <netdb.h>
#include <sys/socket.h>
#include <string.h>
#include <unistd.h>

// open scene graph
#include <osg/Vec3>
#include <osg/Matrixd>

// CalVR plugin support
#include <cvrConfig/ConfigManager.h>

// local includes
#include "OSCPack.h"

class AudioHandler
{
  public:
    AudioHandler();
    ~AudioHandler();

    void connectServer();
    void disconnectServer();
    bool isConnected() { return _isConnected; }
    void loadSound(int soundID, osg::Vec3 &dir, osg::Vec3 &pos);
    void loadSound(int soundID, float az);
    void playSound(int soundID, std::string sound);
    void update(int soundID, const osg::Vec3 &viewDir, const osg::Vec3 &viewPos);
    void update(int soundID, float az);
    
  protected:
    bool _isConnected;
    int _sock;
};


#endif
