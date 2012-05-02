#include "TouchDesigner.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/CVRSocket.h>

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>

#define PORT 19997

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(TouchDesigner)

TouchDesigner::TouchDesigner()
{
}

bool TouchDesigner::init()
{
    std::cerr << "TouchDesigner init\n";

    _menu = new SubMenu("TouchDesigner", "TouchDesigner");
    _menu->setCallback(this);

    _receiveButton = new MenuButton("Receive Data");
    _receiveButton->setCallback(this);
    _menu->addItem(_receiveButton);

    _port = ConfigManager::getEntry("Plugin.TouchDesigner.Port");

    MenuSystem::instance()->addMenuItem(_menu);

    initSocket();

    std::cerr << "TouchDesigner init done.\n";
    return true;
}

void TouchDesigner::initSocket()
{
  if ((_sockID = socket(AF_INET, SOCK_DGRAM, 0)) == -1)
  {
    cerr<<"Socket Error!"<<endl;
    exit(1); 
  }

  _serverAddr.sin_family = AF_INET;
  _serverAddr.sin_port = htons(PORT);
  _serverAddr.sin_addr.s_addr = INADDR_ANY;
  bzero(&(_serverAddr.sin_zero), 8);

  if (bind(_sockID, (struct sockaddr *)&_serverAddr, sizeof(struct sockaddr)) == -1)
  {
    cerr<<"Bind Error!"<<endl;
    exit(1);
  }

  _addrLen = sizeof(struct sockaddr);
  cerr<<"Server waiting for client on port: "<<PORT<<endl;
}

void TouchDesigner::readSocket()
{
    char recvData[1024];
    string str;
    int bytes_read;
    int rs;
    
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(_sockID, &readfds);
//    while(!_mkill)
    {
        rs = select(_sockID + 1, &readfds, 0, 0, 0);
        if(rs > 0){
            bytes_read = recvfrom(_sockID, recvData, 1024 , 0, (struct sockaddr *)&_clientAddr, &_addrLen);
            if(bytes_read <= 0){
                cerr<<"No data read."<<endl;
            }
            // Prepares data for processing...
            recvData[bytes_read]='\0';
            str = recvData;
            cerr << "received: " << str << endl;
        }
    }
}

TouchDesigner::~TouchDesigner()
{
}

void TouchDesigner::preFrame()
{
    int numBytes;
    char* data;
    char c;

    if(ComController::instance()->isMaster())
    {
      readSocket();
//      cerr << "running on master" << endl;
/*
    	ComController::instance()->sendSlaves(&numBytes, sizeof(int));
      data = new char[numBytes];
	// fill data array with data from socket
      ComController::instance()->sendSlaves(data, numBytes * sizeof(char));
*/
    }
    else
    {
/*
    	ComController::instance()->readMaster(&numBytes, sizeof(int));
    	if(numBytes>0)
    	{
	      data = new char[numBytes];
	      ComController::instance()->readMaster(data, numBytes * sizeof(char));
    		// process data array
    	}
*/
    }
}

void TouchDesigner::menuCallback(MenuItem* menuItem)
{
    if(menuItem == _receiveButton)
    {
      receiveGeometry();
      return;
    }
}

void TouchDesigner::receiveGeometry()
{
  cerr << "TouchDesigner::receiveGeometry" << endl;
}


