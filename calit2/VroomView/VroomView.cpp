#include "VroomView.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>

#include <iostream>

using namespace cvr;

CVRPLUGIN(VroomView)

VroomView::VroomView()
{
	_mls = NULL;
	_localSS = NULL;
}

VroomView::~VroomView()
{
	if(_mls)
	{
		delete _mls;
	}
}

bool VroomView::init()
{
	if(ComController::instance()->isMaster())
	{
		int port = ConfigManager::getInt("value","Plugin.VroomView.ListenPort",12121);
		_mls = new MultiListenSocket(port);
		if(!_mls->setup())
		{
			std::cerr << "Error setting up MultiListen Socket on port " << port << " ." << std::endl;
			delete _mls;
			_mls = NULL;
		}
		else
		{
			std::cerr << "Socket listening on port: " << port << std::endl;
		}
		_localSS = new VVLocal();
	}

	return true;
}

void VroomView::preFrame()
{
	if(ComController::instance()->isMaster())
	{
		if(_mls)
		{
			CVRSocket * con;
			while((con = _mls->accept()))
			{
				std::cerr << "Adding socket." << std::endl;
				con->setNoDelay(true);
				_clientList.push_back(new VVClient(con));
			}
		}

		for(int i = 0; i < _clientList.size(); ++i)
		{
		    _clientList[i]->preFrame();
		}

		_localSS->preFrame();

		for(std::vector<VVClient*>::iterator it = _clientList.begin(); it != _clientList.end(); )
		{
		    if((*it)->isError())
		    {
			delete (*it);
			it = _clientList.erase(it);
		    }
		    else
		    {
			it++;
		    }
		}

	}
}

bool VroomView::processEvent(cvr::InteractionEvent * event)
{
    KeyboardInteractionEvent * kie = event->asKeyboardEvent();
    if(kie && kie->getKey() == (int)'N' && kie->getInteraction() == KEY_UP)
    {
	if(ComController::instance()->isMaster())
	{
	    std::cerr << "Print Screen" << std::endl;
	    _localSS->takeScreenShot();
	}
	return true;
    }
    return false;
}
