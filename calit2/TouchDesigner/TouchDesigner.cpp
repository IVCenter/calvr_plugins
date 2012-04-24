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

    std::cerr << "TouchDesigner init done.\n";
    return true;
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
/*
        if((sock = _mls->accept()))
        if(!socket->recv(&c,sizeof(char)))
        {
          cerr << "nothing received" << endl;
	        return false;
        }

        std::cerr << "char: " << c << std::endl;

	checkSockets();

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
	int BUFSIZE = 500;

	cvr::CVRSocket server=cvr::CVRSocket(LISTEN, "128.54.37.189", 6662 ,AF_INET,SOCK_DGRAM);
	cerr << "server created" << endl;
	server.setReuseAddress(true);
	server.bind();
        cerr << "bound" << endl;
		
	char * data = (char *) malloc(sizeof(char)*BUFSIZE);
	bool received=server.recv(data,BUFSIZE);
	if (received){
		cerr << "data received: " << data << endl;
	}

/*
  std::cerr << "TouchDesigner: receiving geometry" << std::endl;
  _sockID = (int)socket(AF_INET,SOCK_STREAM,0);
  if(_sockID == -1)
  {
    cerr << "Error creating socket." << endl;
    return;
  }
  else cerr << "socket created, ID=" << _sockID << endl;

  int yes = 1;
  if(setsockopt(_sockID, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(int)) == -1)
  {
    perror("setsockopt");
    cerr << "Error setting reuseaddress option." << endl;
    return;
  }
  else cerr << "reuseaddress option set" << endl;

  int flags = fcntl(_sockID, F_GETFL, 0);
  fcntl(_sockID, F_SETFL, flags | O_NONBLOCK);

  sockaddr_in addr;
  memset(&addr,0,sizeof(addr));

  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(_port);

  if(bind(_sockID,(struct sockaddr *)&addr,sizeof(addr)) == -1)
  {
      cerr << "Error on socket bind." << endl;
      perror("bind");
      return false;
  }

  if(listen(_sockID,_queue) == -1)
  {
      cerr << "Error on socket listen." << endl;
      return false;
  }
*/
}

