#include "PointsObject.h"
#include "PanMarkerObject.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrConfig/ConfigManager.h>

using namespace cvr;

PointsObject::PointsObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _activePanMarker = NULL;
    _transitionActive = false;
    _transitionTime = 4.0;
}

PointsObject::~PointsObject()
{

}

bool PointsObject::getPanActive()
{
    return _activePanMarker;
}

void PointsObject::setActiveMarker(PanMarkerObject * marker)
{
    _activePanMarker = marker;
    if(marker)
    {
	startTransition();
    }
}

void PointsObject::panUnloaded()
{
    if(_activePanMarker)
    {
	_root->setNodeMask(_storedNodeMask);
	_activePanMarker->panUnloaded();
	_activePanMarker = NULL;
	setNavigationOn(true);
    }
}

void PointsObject::clear()
{
    _root->removeChildren(0,_root->getNumChildren());
    setTransform(osg::Matrix::identity());
}

void PointsObject::update()
{
    if(_transitionActive)
    {
	float lastStatus = _transition / _transitionTime;
	_transition += PluginHelper::getLastFrameDuration();
	if(_transition > _transitionTime)
	{
	    _transition = _transitionTime;
	}

	float status = _transition / _transitionTime;
	status -= lastStatus;

	osg::Vec3 movement = (_startCenter - _endCenter);
	movement.x() *= status;
	movement.y() *= status;
	movement.z() *= status;
	osg::Matrix m;
	m.makeTranslate(movement);

	setTransform(getTransform() * m);

	if(_transition == _transitionTime)
	{
	    _transitionActive = false;
	    if(_activePanMarker->loadPan())
	    {
		_storedNodeMask = _root->getNodeMask();
		_root->setNodeMask(0);
	    }
	    else
	    {
		_activePanMarker = NULL;
		setNavigationOn(true);
	    }
	}
    }
}

void PointsObject::updateCallback(int handID, const osg::Matrix & mat)
{
    osg::Vec3 viewerPos = PluginHelper::getHeadMat().getTrans();
    for(int i = 0; i < getNumChildObjects(); i++)
    {
	osg::Vec3 spherePos = getChildObject(i)->getObjectToWorldMatrix().getTrans();
	float distance = (spherePos - viewerPos).length();
	PanMarkerObject * pmo = dynamic_cast<PanMarkerObject*>(getChildObject(i));
	if(pmo)
	{
	    pmo->setViewerDistance(distance);
	}
    }
}

void PointsObject::startTransition()
{
    //TODO: find to/from movement points
    _transitionActive = true;
    setNavigationOn(false);
    _transition = 0.0;

    osg::Vec3 offset = ConfigManager::getVec3("Plugin.PanoViewLOD.Offset");
    offset = offset + osg::Vec3(0,0,_activePanMarker->getCenterHeight());

    _endCenter = offset;
    _startCenter = _activePanMarker->getObjectToWorldMatrix().getTrans();

}
