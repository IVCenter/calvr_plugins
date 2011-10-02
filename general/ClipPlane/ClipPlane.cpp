#include "ClipPlane.h"

#include <kernel/PluginHelper.h>
#include <kernel/InteractionManager.h>

#include <iostream>
#include <sstream>

CVRPLUGIN(ClipPlane)

using namespace cvr;

ClipPlane::ClipPlane()
{
    _activePlane = -1;
}

ClipPlane::~ClipPlane()
{
    
}

bool ClipPlane::init()
{
    _clipPlaneMenu = new SubMenu("ClipPlane","ClipPlane");
    for(int i = 0; i < MAX_CLIP_PLANES; i++)
    {
	_planeList.push_back(new osg::ClipPlane(i));

	std::stringstream ss;
	ss << "Place Plane " << i;
	MenuCheckbox * cb = new MenuCheckbox(ss.str(),false);
	cb->setCallback(this);
	_placeList.push_back(cb);
	_clipPlaneMenu->addItem(cb);

	std::stringstream ss2;
	ss2 << "Enable Plane " << i;
	cb = new MenuCheckbox(ss2.str(),false);
	cb->setCallback(this);
	_enableList.push_back(cb);
	_clipPlaneMenu->addItem(cb);
    }

    PluginHelper::addRootMenuItem(_clipPlaneMenu);

    return true;
}

void ClipPlane::menuCallback(MenuItem * item)
{
    for(int i = 0; i < _placeList.size(); i++)
    {
	if(item == _placeList[i])
	{
	    if(_placeList[i]->getValue())
	    {
		if(_activePlane >= 0)
		{
		    _placeList[i]->setValue(false);
		}
		_activePlane = i;
		if(!_enableList[i]->getValue())
		{
		    _enableList[i]->setValue(true);
		    menuCallback(_enableList[i]);
		}
		//osg::StateSet * stateset = PluginHelper::getObjectTransform()->getOrCreateStateSet();
    
		//stateset->setAttributeAndModes(_planeList[i],osg::StateAttribute::ON);
	    }
	    else
	    {
		_activePlane = -1;
	    }
	    return;
	}
    }

    for(int i = 0; i < _enableList.size(); i++)
    {
	if(item == _enableList[i])
	{
	    //osg::StateSet * stateset = PluginHelper::getObjectTransform()->getOrCreateStateSet();

	    if(_enableList[i]->getValue())
	    {
		//std::cerr << "Enable plane " << i << std::endl;
		//stateset->setAttributeAndModes(_planeList[i],osg::StateAttribute::ON);
		PluginHelper::getObjectsRoot()->addClipPlane(_planeList[i].get());
	    }
	    else
	    {
		if(i == _activePlane)
		{
		    _placeList[i]->setValue(false);
		    menuCallback(_placeList[i]);
		}
		//std::cerr << "Disable plane " << i << std::endl;
		PluginHelper::getObjectsRoot()->removeClipPlane(_planeList[i].get());
		//stateset->setAttributeAndModes(_planeList[i],osg::StateAttribute::OFF);
	    }

	    return;
	}
    }
}

bool ClipPlane::processEvent(InteractionEvent * event)
{
    if(!event->getEventType() == TRACKED_BUTTON_INTER_EVENT)
    {
	return false;
    }

    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    if(!tie)
    {
	return false;
    }

    if(tie->getHand() == 0 && tie->getButton() == 0 && tie->getInteraction() == BUTTON_DOWN)
    {
	if(_activePlane >= 0)
	{
	    osg::Vec3 point = tie->getTransform().getTrans();
	    osg::Vec3 normal(0,1,0);
	    normal = normal * tie->getTransform();

	    point = point * PluginHelper::getWorldToObjectTransform();
	    normal = normal * PluginHelper::getWorldToObjectTransform();

	    normal = normal - point;

	    osg::Plane plane(normal, point);

	    _planeList[_activePlane]->setClipPlane(plane);

	    _placeList[_activePlane]->setValue(false);
	    menuCallback(_placeList[_activePlane]);

	    return true;
	}
    }
    return false;
}

void ClipPlane::preFrame()
{
    if(_activePlane >= 0)
    {
	osg::Vec3 point = PluginHelper::getHandMat(0).getTrans();
	osg::Vec3 normal(0,1,0);
	normal = normal * PluginHelper::getHandMat(0);
	
	point = point * PluginHelper::getWorldToObjectTransform();
	normal = normal * PluginHelper::getWorldToObjectTransform();

	normal = normal - point;
	
	osg::Plane plane(normal, point);

	_planeList[_activePlane]->setClipPlane(plane);
	
    }
}
