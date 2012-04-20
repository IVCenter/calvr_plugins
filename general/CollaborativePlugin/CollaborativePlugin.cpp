#include "CollaborativePlugin.h"

#include <cvrMenu/MenuSystem.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrCollaborative/CollaborativeManager.h>

#include <iostream>

using namespace cvr;

CVRPLUGIN(Collaborative)

Collaborative::Collaborative()
{
    _myCB = NULL;
}

Collaborative::~Collaborative()
{
}

bool Collaborative::init()
{
    _collabMenu = new SubMenu("Collaborative","Collaborative");
    _collabMenu->setCallback(this);

    _enable = new MenuCheckbox("Enable",false);
    _enable->setCallback(this);

    _collabMenu->addItem(_enable);

    MenuSystem::instance()->addMenuItem(_collabMenu);

    _server = ConfigManager::getEntry("Plugin.CollaborativePlugin.Server");
    _port = ConfigManager::getInt("Plugin.CollaborativePlugin.Port",11050);

    _lockedCB = new MenuCheckbox("Locked View",false);
    _lockedCB->setCallback(this);

    _clientText = new MenuText("Clients:",1.0,false);

    return true;
}

void Collaborative::menuCallback(MenuItem * item)
{
    if(item == _enable)
    {
	if(_enable->getValue())
	{
	    if(!CollaborativeManager::instance()->connect(_server,_port))
	    {
		_enable->setValue(false);
	    }
	    else
	    {
		addMenuItems();
	    }
	}
	else
	{
	    CollaborativeManager::instance()->disconnect();
	    removeMenuItems();
	}
    }

    if(!_enable->getValue())
    {
	return;
    }

    if(item == _lockedCB)
    {
	if(_lockedCB->getValue())
	{
	    CollaborativeManager::instance()->setMode(LOCKED);
	}
	else
	{
	    CollaborativeManager::instance()->setMode(UNLOCKED);
	}
	return;
    }

    if(item == _myCB)
    {
	if(_myCB->getValue())
	{
	    CollaborativeManager::instance()->setMasterID(CollaborativeManager::instance()->getID());
	}
	else
	{
	    CollaborativeManager::instance()->setMasterID(-1);
	}
	return;
    }

    for(std::map<int,cvr::MenuCheckbox*>::iterator it = _clientCBMap.begin(); it != _clientCBMap.end(); it++)
    {
	if(item == it->second)
	{
	    if(it->second->getValue())
	    {
		CollaborativeManager::instance()->setMasterID(it->first);
	    }
	    else
	    {
		CollaborativeManager::instance()->setMasterID(-1);
	    }
	    return;
	}
    }
}

void Collaborative::preFrame()
{
    if(_enable->getValue() != CollaborativeManager::instance()->isConnected())
    {
	_enable->setValue(CollaborativeManager::instance()->isConnected());
	if(CollaborativeManager::instance()->isConnected())
	{
	    addMenuItems();
	}
	else
	{
	    removeMenuItems();
	}
	return;
    }

    if(!_enable->getValue())
    {
	return;
    }

    updateMenuItems();
}

void Collaborative::addMenuItems()
{
    _collabMenu->addItem(_lockedCB);
    _collabMenu->addItem(_clientText);

    if(!_myCB)
    {
	_myCB = new MenuCheckbox(CollaborativeManager::instance()->getName(),false);
	_myCB->setCallback(this);
    }

    _collabMenu->addItem(_myCB);

    updateMenuItems();
}

void Collaborative::removeMenuItems()
{
    _collabMenu->removeItem(_lockedCB);
    _collabMenu->removeItem(_clientText);

    if(_myCB)
    {
	_collabMenu->removeItem(_myCB);
    }

    for(std::map<int,cvr::MenuCheckbox*>::iterator it = _clientCBMap.begin(); it != _clientCBMap.end(); it++)
    {
	_collabMenu->removeItem(it->second);
	delete it->second;
    }
    _clientCBMap.clear();
}

void Collaborative::updateMenuItems()
{
    _lockedCB->setValue(CollaborativeManager::instance()->getMode() == LOCKED);

    std::vector<int> removeList;

    for(std::map<int,cvr::MenuCheckbox*>::iterator it = _clientCBMap.begin(); it != _clientCBMap.end(); it++)
    {
	if(CollaborativeManager::instance()->getClientInitMap().find(it->first) == CollaborativeManager::instance()->getClientInitMap().end())
	{
	    removeList.push_back(it->first);
	}
    }

    for(int i = 0; i < removeList.size(); i++)
    {
	_collabMenu->removeItem(_clientCBMap[removeList[i]]);
	delete _clientCBMap[removeList[i]];
	_clientCBMap.erase(removeList[i]);
    }

    for(std::map<int,ClientInitInfo>::iterator it = CollaborativeManager::instance()->getClientInitMap().begin(); it != CollaborativeManager::instance()->getClientInitMap().end(); it++)
    {
	if(_clientCBMap.find(it->first) == _clientCBMap.end())
	{
	    _clientCBMap[it->first] = new MenuCheckbox(it->second.name,false);
	    _clientCBMap[it->first]->setCallback(this);
	    _collabMenu->addItem(_clientCBMap[it->first]);
	}
    }

    if(_myCB)
    {
	if(_myCB->getValue() && CollaborativeManager::instance()->getID() != CollaborativeManager::instance()->getMasterID())
	{
	    _myCB->setValue(false);
	}
	else if(!_myCB->getValue() && CollaborativeManager::instance()->getID() == CollaborativeManager::instance()->getMasterID())
	{
	    _myCB->setValue(true);
	}
    }

    for(std::map<int,cvr::MenuCheckbox*>::iterator it = _clientCBMap.begin(); it != _clientCBMap.end(); it++)
    {
	if(it->second->getValue() && it->first != CollaborativeManager::instance()->getMasterID())
	{
	    it->second->setValue(false);
	}
	else if(!it->second->getValue() && it->first == CollaborativeManager::instance()->getMasterID())
	{
	    it->second->setValue(true);
	}
    }
}
