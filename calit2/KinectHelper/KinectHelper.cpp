#include "KinectHelper.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/SceneManager.h>

#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace cvr;

CVRPLUGIN(KinectHelper)

KinectHelper::KinectHelper()
{
}

KinectHelper::~KinectHelper()
{
}

bool KinectHelper::init()
{
    std::string server, port;
    bool serverFound, portFound;

    server = ConfigManager::getEntry("value","Plugin.KinectHelper.Server","",&serverFound);
    port = ConfigManager::getEntry("value","Plugin.KinectHelper.Port","",&portFound);

    if(!serverFound)
    {
	std::cerr << "KinectHelper Init Error: no server value in config file." << std::endl;
	return false;
    }

    if(!portFound)
    {
	std::cerr << "KinectHelper Init Warning: no port value in config file, using 9001" << std::endl;
	port = "9001";
    }

    _ncCMD = "nc " + server + " " + port;

    _state = NO_INTERACTION;

    sendState();

    return true;
}

void KinectHelper::preFrame()
{
    InteractionState newState = NO_INTERACTION;

    if(Navigation::instance()->getEventActive())
    {
	if(Navigation::instance()->getPrimaryButtonMode() == MOVE_WORLD)
	{
	    newState = PICKING_INTERACTION;
	}
	else if(Navigation::instance()->getPrimaryButtonMode() != NONE)
	{
	    newState = NAVIGATION_INTERACTION;
	}
    }
    else if(SceneManager::instance()->getMenuRoot()->getNumChildren())
    {
	newState = MENU_INTERACTION;
    }

    if(_state != newState)
    {
	_state = newState;
	sendState();
    }
}

void KinectHelper::sendState()
{
    if(ComController::instance()->isMaster())
    {
	std::stringstream ss;
	ss << "echo " << (int)_state << " | " << _ncCMD;
	system(ss.str().c_str());
    }
}
