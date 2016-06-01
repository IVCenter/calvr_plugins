#include "ShadowObject.h"

using namespace cvr;

ShadowObject::ShadowObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,true,showBounds)
{
    _active = false;

    _resetButton = new MenuButton("Reset");
    _resetButton->setCallback(this);
    addMenuItem(_resetButton);
}

ShadowObject::~ShadowObject()
{
}

void ShadowObject::preFrame()
{
    bool active = isActive();

    if(_moving && !active)
    {
	_activeLock.lock();
	_active = true;
	_activeLock.unlock();
    }
}

void ShadowObject::menuCallback(MenuItem * item)
{
    if(item == _resetButton)
    {
	_activeLock.lock();
	_active = false;
	_activeLock.unlock();
	return;
    }

    SceneObject::menuCallback(item);
}
