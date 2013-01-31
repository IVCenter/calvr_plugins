#include "PDObject.h"

#include <cvrKernel/NodeMask.h>

using namespace cvr;

PDObject::PDObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    //_root->setNodeMask(_root->getNodeMask() & ~(CULL_ENABLE));
    setBoundsCalcMode(MANUAL);
    setBoundingBox(osg::BoundingBox(osg::Vec3(-100000,-100000,-100000),osg::Vec3(100000,100000,100000)));
}

PDObject::~PDObject()
{
}
