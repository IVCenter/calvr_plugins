#include "CollaborativePlugin.h"

#include <menu/MenuSystem.h>
#include <config/ConfigManager.h>
#include <collaborative/CollaborativeManager.h>

#include <iostream>

using namespace cvr;

CVRPLUGIN(Collaborative)

Collaborative::Collaborative()
{
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
	}
	else
	{
	    CollaborativeManager::instance()->disconnect();
	}
    }
}
