#include "WoundVAC.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>

#include <iostream>

using namespace cvr;

CVRPLUGIN(WoundVAC)

WoundVAC::WoundVAC()
{
    _socket = NULL;
    _menu = NULL;
    _connectCB = NULL;
    _simObject = NULL;
}

WoundVAC::~WoundVAC()
{
    if(_socket)
    {
	delete _socket;
    }
}

bool WoundVAC::init()
{
    _menu = new SubMenu("WoundVAC");

    _connectCB = new MenuCheckbox("Connect",false);
    _connectCB->setCallback(this);
    _menu->addItem(_connectCB);

    PluginHelper::addRootMenuItem(_menu);

    _host = ConfigManager::getEntry("value","Plugin.WoundVAC.Host","137.110.118.145");
    _port = ConfigManager::getInt("value","Plugin.WoundVAC.Port",8765);

    return true;
}

void WoundVAC::preFrame()
{
    if(_simObject)
    {
	_simObject->preFrame();
    }
}

void WoundVAC::menuCallback(MenuItem * item)
{
    if(item == _connectCB)
    {
	if(_connectCB->getValue())
	{
	    if(_socket)
	    {
		delete _simObject;
		_simObject = NULL;
		delete _socket;
		_socket = NULL;
	    }

	    _socket = new CVRSocket(cvr::CONNECT,_host,_port);
	    _socket->setNoDelay(true);
	    if(!_socket->connect(1))
	    {
		//std::cerr << "Error connecting to device " << _host << ":" << _port << std::endl;
		_connectCB->setValue(false);
		delete _socket;
		_socket = NULL;
		return;
	    }

	    _simObject = new SimulationObject(_socket,"Simulation",true,false,false,true,false);
	    PluginHelper::registerSceneObject(_simObject,"WoundVAC");
	    _simObject->attachToScene();

	    /*int size = 0;
	    _socket->recv(&size,sizeof(int));

	    std::cerr << "Data Length: " << size << std::endl;

	    char * data = new char[size];

	    _socket->recv(data,size);

	    std::cerr << "Data: " << data << std::endl;

	    std::string dataStr = data;

	    delete[] data;

	    char c;
	    _socket->recv(&c,sizeof(char));

	    std::cerr << "Got signal: " << (int)c << std::endl;
	    size = dataStr.size() + 1;
	    _socket->send(&size,sizeof(int));
	    _socket->send(&dataStr[0],size);

	    std::cerr << "sent" << std::endl;
	    _socket->recv(&c,sizeof(char));
	    std::cerr << "Got signal: " << (int)c << std::endl;
	    _socket->send(&size,sizeof(int));
	    _socket->send(&dataStr[0],size);*/
	    
	}
	else
	{
	    if(_simObject)
	    {
		delete _simObject;
		_simObject = NULL;
	    }
	    if(_socket)
	    {
		delete _socket;
		_socket = NULL;
	    }
	}

	return;
    }
}
