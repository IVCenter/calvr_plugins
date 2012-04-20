#include "PanMarkerObject.h"
#include "PointsObject.h"

#include <kernel/PluginHelper.h>
#include <PluginMessageType.h>

#include <osg/CullFace>

using namespace cvr;

PanMarkerObject::PanMarkerObject(float scale, float rotationOffset, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _viewerInRange = false;
    _scale = scale;
    _rotationOffset = rotationOffset;
    osg::Sphere * sphere = new osg::Sphere(osg::Vec3(0,0,0),250.0/_scale);
    _sphere = new osg::ShapeDrawable(sphere);
    _sphere->setColor(osg::Vec4(1.0,0,0,0.5));
    _sphereGeode = new osg::Geode();
    _sphereGeode->addDrawable(_sphere);
    addChild(_sphereGeode);

    osg::StateSet * stateset = _sphereGeode->getOrCreateStateSet();
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);

    osg::CullFace * cf = new osg::CullFace();
    stateset->setAttributeAndModes(cf,osg::StateAttribute::ON);
}

PanMarkerObject::~PanMarkerObject()
{
}

bool PanMarkerObject::processEvent(InteractionEvent * ie)
{
    TrackedButtonInteractionEvent * tie = ie->asTrackedButtonEvent();
    if(tie)
    {
	if(!tie->getButton() && tie->getInteraction() == BUTTON_DOWN && _viewerInRange)
	{
	    if(_parent)
	    {
		PointsObject * po = dynamic_cast<PointsObject*>(_parent);
		if(po && !po->getPanActive())
		{	    
		    std::cerr << "Pan Transition." << std::endl;
		    po->setActiveMarker(this);
		}
	    }
	}
    }

    return SceneObject::processEvent(ie);
}

void PanMarkerObject::setViewerDistance(float distance)
{
    if(distance < 2500.0)
    {
	_sphere->setColor(osg::Vec4(0.0,1.0,0.0,0.5));
	_viewerInRange = true;
    }
    else
    {
	_sphere->setColor(osg::Vec4(1.0,0.0,0.0,0.5));
	_viewerInRange = false;
    }
}

bool PanMarkerObject::loadPan()
{
    PanLoadRequest plr;
    plr.name = getName();
    plr.rotationOffset = _rotationOffset;
    plr.plugin = "PointsWithPans";
    plr.pluginMessageType = PWP_PAN_UNLOADED;
    plr.loaded = false;

    PluginHelper::sendMessageByName("PanoViewLOD",PAN_LOAD_REQUEST,(char*)&plr);

    if(plr.loaded)
    {
	removeChild(_sphereGeode);
    }
    else
    {
	return false;
    }

    return true;
}

void PanMarkerObject::panUnloaded()
{
    if(!_sphereGeode->getNumParents())
    {
	addChild(_sphereGeode);
    }
}
