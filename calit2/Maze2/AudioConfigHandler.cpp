/***************************************************************
* File Name: AudioConfigHandler.cpp
*
* Description: Implementation of audio effects controller
*
* Written by ZHANG Lelin on May 18, 2011
*
***************************************************************/
#include "AudioConfigHandler.h"


using namespace osg;
using namespace std;
using namespace cvr;


// Constructor
AudioConfigHandler::AudioConfigHandler(): mFlagConnected(false), mFlagMaster(false)
{
}


// Destructor
AudioConfigHandler::~AudioConfigHandler()
{
    if (mSockfd) close(mSockfd);
}


/***************************************************************
* Function: connectServer()
*
* In the plugin 'Maze2', audion configurator get the server IP
* from the content stored in 'Maze2'.
*
***************************************************************/
void AudioConfigHandler::connectServer()
{
    if (!mFlagMaster) return;

    /* check with configuration file to decide use audio server or not */
    string useASStr = ConfigManager::getEntry("Plugin.Maze2.AudioServer");
    if (useASStr.compare("on")) return;

    /* get server address and port number */
    string server_addr = ConfigManager::getEntry("Plugin.Maze2.AudioServerAddress");
    int port_number = ConfigManager::getInt("Plugin.Maze2.AudioServerPort", 15003);

    if( server_addr == "" ) server_addr = "127.0.0.1";

    /* build up socket communications */
    int portno = port_number, protocol = SOCK_DGRAM;
    struct sockaddr_in server;
    struct hostent *hp;

    mSockfd = socket(AF_INET, protocol, 0);
    if (mSockfd < 0)
    {
        cerr << "Maze2::connectServer(): Can't open socket for audio server." << endl;
        return;
    }

    server.sin_family = AF_INET;
    hp = gethostbyname(server_addr.c_str());
    if (hp == 0)
    {
        cerr << "Maze2::connectServer(): Unknown audio server host." << endl;
        close(mSockfd);
        return;
    }
    memcpy((char *)&server.sin_addr, (char *)hp->h_addr, hp->h_length);
    server.sin_port = htons((unsigned short)portno);

    /* connect to audio server */ 
    if (connect(mSockfd, (struct sockaddr *) &server, sizeof (server)) < 0)
    {
	cerr << "Maze2::connect(): Failed connect to server" << endl;
        close(mSockfd);
        return;
    }

    cerr << "Maze2::AudioConfigHandler::Successfully connected to Audio Server." << endl;

    mFlagConnected = true;
}


/***************************************************************
* Function: disconnectServer()
***************************************************************/
void AudioConfigHandler::disconnectServer()
{
    if (!mFlagMaster) return;

    close(mSockfd);
    mFlagConnected = false;
    mSockfd = 0;
}


/***************************************************************
* Function: loadSoundSource()
***************************************************************/
void AudioConfigHandler::loadSoundSource()
{
    uint8_t packet[256];
    int32_t size;
    
    size = oscpack(packet, "/sc.environment/model", "s", "MediumModel.MAZ");
    if (send(mSockfd, (char *) packet, size, 0))
    {
        // DEBUG MSG: 
        cerr << "AudioConfigHandler  Model info: " << "MediumModel.MAZ" << endl;
    }
    
    size = oscpack(packet, "/sc.environment/pose", "iffffff", 1, 9.0f, 9.0f, 0.0f, M_PI * 0.5f, M_PI * 0.5f, 0.0f);
    if (send(mSockfd, (char *) packet, size, 0))
    {
        // DEBUG MSG: 
        cerr << "AudioConfigHandler  Sound's info: " << 0 << " " << 0 << " " << 0 << endl;
    }
}

/***************************************************************
* Function: updateGeometry()
***************************************************************/
void AudioConfigHandler::updateGeometry()
{
    if (!(mFlagConnected && mFlagMaster)) return;
}


/***************************************************************
* Function: updatePoses()
*
* Send viewer & sources info via OSC/UDP to audio server.
* 
* - CAVEDesigner::update()
*
***************************************************************/
void AudioConfigHandler::updatePoses(const osg::Vec3 &viewDir, const osg::Vec3 &viewPos)
{
    if (!(mFlagConnected && mFlagMaster)) return;

    uint8_t packet[256];
    int32_t size;

    /* OSCPack Messenger: send pose info within every frame, typ = 0 for observer */
    int typ = 0;
    float yaw = acos(viewDir.x());
    if (viewDir.y() < 0) yaw = M_PI * 2 - yaw;
    yaw += M_PI * 0.5;
    if (yaw > M_PI * 2) yaw -= M_PI * 2;
    float pitch = M_PI * 0.5 - atan(viewDir.z());
    float roll = 0.0f;
    size = oscpack(packet, "/sc.environment/pose", "iffffff", typ, viewPos.x(), viewPos.y(), viewPos.z(), yaw, pitch, roll);
    if (send(mSockfd, (char *) packet, size, 0))
    {
	// DEBUG MSG: 
	cerr << "AudioConfigHandler  Viewer's info: " << 
	// DEBUG MSG:
	viewPos.x() << " " << viewPos.y() << " " << viewPos.z() << endl;	
    }

    /* OSCPack Messenger: update others for sound sources here, typ > 0 for sound sources */
    /* testing sound source 
    size = oscpack(packet, "/sc.environment/pose", "iffffff", 1, 9.0f, 9.0f, 0.0f, M_PI * 0.5f, M_PI * 0.5f, 0.0f);
    if (send(mSockfd, (char *) packet, size, 0))
    {
        // DEBUG MSG: cerr << "AudioConfigHandler  Sound's info: " << 0 << " " << 0 << " " << 0 << endl;
    }
    */
}







