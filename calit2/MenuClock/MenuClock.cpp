#include "MenuClock.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/ComController.h>

#include <ctime>
#include <iostream>

using namespace cvr;

CVRPLUGIN(MenuClock)

MenuClock::MenuClock()
{
}

MenuClock::~MenuClock()
{
}

bool MenuClock::init()
{
    _itemGroup = new MenuItemGroup();
    _clockText = new MenuText("");
    _clockText->setIndent(false);
    _itemGroup->addItem(_clockText);
    
    MenuSystem::instance()->getMenu()->addItem(_itemGroup,0);

    updateClock();

    return true;
}

void MenuClock::preFrame()
{
    updateClock();
}

void MenuClock::updateClock()
{
    char newTime[128];

    if(ComController::instance()->isMaster())
    {
	time_t now = time(NULL);
	strftime(newTime,128,"%R",localtime(&now));

	ComController::instance()->sendSlaves(newTime,128*sizeof(char));
    }
    else
    {
	ComController::instance()->readMaster(newTime,128*sizeof(char));
    }
    
    std::string ntimestr = newTime;
    if(ntimestr != _clockStr)
    {
	_clockStr = ntimestr;
	_clockText->setText(_clockStr);
    }
}
