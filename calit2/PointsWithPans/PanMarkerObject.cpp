#include "PanMarkerObject.h"
#include "PointsObject.h"
#include "TexturedSphere.h"

#include <cvrKernel/PluginHelper.h>
#include <PluginMessageType.h>

#include <osg/CullFace>

using namespace cvr;

PanMarkerObject::PanMarkerObject(float scale, float rotationOffset, float radius, float selectDistance, std::string name, std::string textureFile, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _viewerInRange = false;
    _scale = scale;
    _rotationOffset = rotationOffset;

    if(textureFile.empty())
    {
	osg::Sphere * sphere = new osg::Sphere(osg::Vec3(0,0,0),radius/_scale);
	_sphere = new osg::ShapeDrawable(sphere);
	_sphere->setColor(osg::Vec4(1.0,0,0,0.5));
	_sphereGeode = new osg::Geode();
	_sphereGeode->addDrawable(_sphere);

	osg::StateSet * stateset = _sphereGeode->getOrCreateStateSet();
	std::string bname = "spheres";
	stateset->setRenderBinDetails(2,bname);
	stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    }
    else
    {
	_sphereGeode = TexturedSphere::makeSphere(textureFile,radius/_scale,2.0);
    }

    addChild(_sphereGeode);

    _selectDistance = selectDistance;

    _name = name;

    PanHeightRequest phr;
    phr.name = name;
    phr.height = 0.0;

    PluginHelper::sendMessageByName("PanoViewLOD",PAN_HEIGHT_REQUEST,(char*)&phr);
    _centerHeight = phr.height;

    osg::StateSet * stateset = _sphereGeode->getOrCreateStateSet();

    osg::CullFace * cf = new osg::CullFace();
    stateset->setAttributeAndModes(cf,osg::StateAttribute::ON);

    addMoveMenuItem();
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
	    if(_parent && !getMovable())
	    {
		PointsObject * po = dynamic_cast<PointsObject*>(_parent);
		if(po && !po->getPanActive())
		{	    
		    std::cerr << "Pan Transition." << std::endl;
		    po->setActiveMarker(this);
		    return true;
		}
	    }
	}
    }

    return SceneObject::processEvent(ie);
}

void PanMarkerObject::enterCallback(int handID, const osg::Matrix &mat)
{
    osg::Vec3 pos = getTransform().getTrans();
    std::cerr << "Marker " << _name << ": x: " << pos.x() << " y: " << pos.y() << " z: " << pos.z() << std::endl;
}

void PanMarkerObject::setViewerDistance(float distance)
{
    if(distance < _selectDistance)
    {
	if(_sphere)
	{
	    _sphere->setColor(osg::Vec4(0.0,1.0,0.0,0.5));
	}
	_viewerInRange = true;
    }
    else
    {
	if(_sphere)
	{
	    _sphere->setColor(osg::Vec4(1.0,0.0,0.0,0.5));
	}
	_viewerInRange = false;
    }
}

bool PanMarkerObject::loadPan()
{
    osg::Vec3 dir(0,1,0);
    dir = dir * getWorldToObjectMatrix();
    dir = dir - getWorldToObjectMatrix().getTrans();
    dir.z() = 0;
    dir.normalize();
    osg::Matrix m;
    osg::Vec3 axis(0,1,0);
    double rot;
    m.makeRotate(dir,axis);
    m.getRotate().getRotate(rot,axis);
    if(axis.z() < 0)
    {
        rot = (2.0 * M_PI) - rot;
    }

    std::cerr << "Nav rotation offset: " << rot * 180.0 / M_PI << std::endl;
    _currentRotation = rot;

    PanLoadRequest plr;
    plr.name = getName();
    plr.rotationOffset = rot + _rotationOffset;
    plr.plugin = "PointsWithPans";
    plr.pluginMessageType = PWP_PAN_UNLOADED;
    plr.loaded = false;

    PluginHelper::sendMessageByName("PanoViewLOD",PAN_LOAD_REQUEST,(char*)&plr);

    if(plr.loaded)
    {
	//removeChild(_sphereGeode);
    }
    else
    {
	return false;
    }

    return true;
}

void PanMarkerObject::panUnloaded()
{
    /*if(!_sphereGeode->getNumParents())
    {
	addChild(_sphereGeode);
    }*/
}

void PanMarkerObject::hide()
{
    removeChild(_sphereGeode);
}

void PanMarkerObject::unhide()
{
    if(!_sphereGeode->getNumParents())
    {
	addChild(_sphereGeode);
    }
}
