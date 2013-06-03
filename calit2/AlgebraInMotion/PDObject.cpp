#include "PDObject.h"

#include <cvrKernel/NodeMask.h>
#include <cvrKernel/Navigation.h>

using namespace cvr;

PDObject::PDObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    //_root->setNodeMask(_root->getNodeMask() & ~(CULL_ENABLE));
    setBoundsCalcMode(MANUAL);
    setBoundingBox(osg::BoundingBox(osg::Vec3(-100000,-100000,-100000),osg::Vec3(100000,100000,100000)));

    _resetPositionButton = NULL;
    if(contextMenu)
    {
	_resetPositionButton = new MenuButton("Reset Position");
	_resetPositionButton->setCallback(this);
	addMenuItem(_resetPositionButton);
    }
}

PDObject::~PDObject()
{
    if(_resetPositionButton)
    {
	delete _resetPositionButton;
    }
}

void PDObject::menuCallback(MenuItem * item)
{
    if(item == _resetPositionButton)
    {
	resetPosition();
	return;
    }

    SceneObject::menuCallback(item);
}

void PDObject::resetPosition()
{
    setNavigationOn(false);
    osg::Matrix m, ms, mt;
    m.makeRotate((90.0/180.0)*M_PI,osg::Vec3(1.0,0,0));
    ms.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
    mt.makeTranslate(osg::Vec3(0,0,-Navigation::instance()->getFloorOffset()));
    setTransform(m*ms*mt);
    setNavigationOn(true);
}
