/***************************************************************
* File Name: AudioConfigHandler.cpp
*
* Description: Implementation of audio effects controller
*
* Written by ZHANG Lelin on Mar 9, 2011
*
***************************************************************/
#include "AudioConfigHandler.h"


using namespace cvr;
using namespace std;
using namespace osg;


// Constructor
AudioConfigHandler::AudioConfigHandler(Switch *shapeSwitchPtr): mFlagConnected(false), mFlagMaster(false),
		mCAVEShapeSwitch(shapeSwitchPtr)
{
}


// Destructor
AudioConfigHandler::~AudioConfigHandler()
{
    if (mSockfd) close(mSockfd);
}


/***************************************************************
* Function: connectServer()
***************************************************************/
void AudioConfigHandler::connectServer()
{
    if (!mFlagMaster) return;

    /* check with configuration file to decide use audio server or not */
    string useASStr = ConfigManager::getEntry("Plugin.CaveCADBeta.AudioServer");
    if (useASStr.compare("on")) return;

    /* get server address and port number */
    string server_addr = ConfigManager::getEntry("Plugin.CaveCADBeta.AudioServerAddress");
    int port_number = ConfigManager::getInt("Plugin.CaveCADBeta.AudioServerPort", 8084);

    if( server_addr == "" ) server_addr = "127.0.0.1";

    /* build up socket communications */
    int portno = port_number, protocol = SOCK_DGRAM;
    struct sockaddr_in server;
    struct hostent *hp;

    mSockfd = socket(AF_INET, protocol, 0);
    if (mSockfd < 0)
    {
        cerr << "CaveCADBeta::connectServer(): Can't open socket for audio server." << endl;
        return;
    }

    server.sin_family = AF_INET;
    hp = gethostbyname(server_addr.c_str());
    if (hp == 0)
    {
        cerr << "CaveCADBeta::connectServer(): Unknown audio server host." << endl;
        close(mSockfd);
        return;
    }
    memcpy((char *)&server.sin_addr, (char *)hp->h_addr, hp->h_length);
    server.sin_port = htons((unsigned short)portno);

    /* connect to audio server */ 
    if (connect(mSockfd, (struct sockaddr *) &server, sizeof (server)) < 0)
    {
	cerr << "CaveCADBeta::connect(): Failed connect to server" << endl;
        close(mSockfd);
        return;
    }

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
* Function: updateShapes()
*
* Collect CAVE geometry info from root node 'mCAVEShapeSwitch',
* sound parameters associated with materials, and send them via
* OSC/UDP to audio server. This function is called each time a
* new geometry is created or modified in the following instances:
*
* - DSGeometryCreator::inputDevReleaseEvent()
* - DSGeometryEditor::inputDevReleaseEvent()
* - DSTexturePallette::inputDevPressEvent()
*
***************************************************************/
void AudioConfigHandler::updateShapes()
{
    if (!(mFlagConnected && mFlagMaster)) return;

    uint8_t packet[256];
    int32_t size;
    string msg;

    /* generate audio config message by iterating children of 'mCAVEShapeSwitch' */
    const int numShapeGroups = mCAVEShapeSwitch->getNumChildren();
    if (numShapeGroups > 0)
    {
	/* index count all geodes approximated 'boxes' */
	int boxCnt = 0;
	for (int i = 0; i < numShapeGroups; i++)
	{
	    CAVEGroupShape *groupShapePtr = dynamic_cast <CAVEGroupShape*> (mCAVEShapeSwitch->getChild(i));
	    if (groupShapePtr)
	    {
		const int numShapeGeodes = groupShapePtr->getNumCAVEGeodeShapes();
		if (numShapeGeodes > 0)
		{
		    for (int j = 0; j < numShapeGeodes; j++)
		    {
			CAVEGeodeShape *geodeShapePtr = groupShapePtr->getCAVEGeodeShape(j);
			const BoundingBox& bb = geodeShapePtr->getBoundingBox();

			/* OSCPack Messenger: send geode bounding box info */
			char boxname[64];
			sprintf(boxname, "box%d", boxCnt++);
			msg = string("/sc.environment/") + string(boxname) + string("/griddef");
			Vec3 dim = Vec3(bb.xMax()-bb.xMin(), bb.yMax()-bb.yMin(), bb.zMax()-bb.zMin());
			size = oscpack(packet, msg.c_str(), "ffffff", bb.xMin(), bb.yMin(), bb.zMin(), 
					dim.x(), dim.y(), dim.z());

			if (send(mSockfd, (char *) packet, size, 0))
			{ 
			    // DEBUG MSG: cerr << "AudioConfigHandler Box: " << dim.x() << " " << dim.y() << " " << dim.z();
			}

			/* OSCPack Messenger: send geode audio material info */
			string materialname = geodeShapePtr->getAudioInfo();
			msg = string("/sc.environment/") + string(boxname) + string("/set/material/all");
			size = oscpack(packet, msg.c_str(), "s", materialname.c_str());
			if (send(mSockfd, (char *) packet, size, 0))
			{
			    // DEBUG MSG: cerr << " " << materialname << endl;
			}
		    }
		}
	    }
	}
    } 
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
    float pitch = M_PI * 0.5 - atan(viewDir.z());
    float roll = 0.0f;
    size = oscpack(packet, "/sc.environment/pose", "iffffff", typ, viewPos.x(), viewPos.y(), viewPos.z(), yaw, pitch, roll);
    if (send(mSockfd, (char *) packet, size, 0))
    {
	// DEBUG MSG: cerr << "AudioConfigHandler  Viewer's info: " << 
	// DEBUG MSG: viewPos.x() << " " << viewPos.y() << " " << viewPos.z() << endl;	
    }

    /* OSCPack Messenger: update others for sound sources here, typ > 0 for sound sources */
    /* testing sound source */
    size = oscpack(packet, "/sc.environment/pose", "iffffff", 1, 0.0f, 0.0f, 0.0f, M_PI * 0.5f, M_PI * 0.5f, 0.0f);
    if (send(mSockfd, (char *) packet, size, 0))
    {
        // DEBUG MSG: cerr << "AudioConfigHandler  Sound's info: " << 0 << " " << 0 << " " << 0 << endl;
    }

}








