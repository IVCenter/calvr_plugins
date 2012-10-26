#include "PointsObject.h"
#include "PanMarkerObject.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrConfig/ConfigManager.h>
#include <PluginMessageType.h>

using namespace cvr;

PointsObject::PointsObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _activePanMarker = NULL;
    _transitionActive = false;
    _fadeActive = false;
    _fadeInActive = false;
    _transitionTime = 4.0;

    _totalFadeTime = 5.0;

    if(contextMenu)
    {
	_alphaRV = new MenuRangeValue("Alpha",0.0,1.0,1.0);
	_alphaRV->setCallback(this);
	addMenuItem(_alphaRV);
    }
    else
    {
	_alphaRV = NULL;
    }

    _root->getOrCreateStateSet()->setMode(GL_BLEND,osg::StateAttribute::ON);
    std::string bname = "StateSortedBin";
    _root->getOrCreateStateSet()->setRenderBinDetails(11,bname);
    _root->getOrCreateStateSet()->setNestRenderBins(false);
    //_root->getOrCreateStateSet()->setBinNumber(1);
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

void PointsObject::panUnloaded(float rotation)
{
    if(_activePanMarker)
    {
	float roffset = rotation - (_activePanMarker->getRotationOffset() + _activePanMarker->getCurrentRotation());
	osg::Vec3 rpoint = _activePanMarker->getObjectToWorldMatrix().getTrans();
        osg::Matrix m;
        m.makeRotate(roffset,osg::Vec3(0,0,1));
        setTransform(getTransform() * osg::Matrix::translate(-rpoint) * m * osg::Matrix::translate(rpoint));

	//_root->setNodeMask(_storedNodeMask);
	if(_alphaUni)
	{
	    _alphaUni->set(0.0f);
	}
	attachToScene();
	_fadeInActive = true;
	_fadeActive = false;
	float panAlpha = 1.0f;
	PluginHelper::sendMessageByName("PanoViewLOD",PAN_SET_ALPHA,(char*)&panAlpha);
	_fadeTime = 0.0;
	/*_activePanMarker->panUnloaded();
	_activePanMarker = NULL;
	setNavigationOn(true);

	for(int i = 0; i < getNumChildObjects(); i++)
	{
	    PanMarkerObject * pmo = dynamic_cast<PanMarkerObject*>(getChildObject(i));
	    if(pmo)
	    {
		pmo->unhide();
	    }
	}*/
    }
}

void PointsObject::clear()
{
    while(getNumChildNodes())
    {
	removeChild(getChildNode(0));
    }
    setTransform(osg::Matrix::identity());
    _alphaUni = NULL;
    _activePanMarker = NULL;
    _transitionActive = false;
    _fadeActive = false;
    _fadeInActive = false;
    _transitionTime = 4.0;
    _totalFadeTime = 5.0;

    while(getNumChildObjects())
    {
	SceneObject * so = getChildObject(0);
	removeChild(so);
	// TODO: fix nested delete
	//delete so;
    }

    detachFromScene();
    PluginHelper::sendMessageByName("PanoViewLOD",PAN_UNLOAD,NULL);
    PluginHelper::sendMessageByName("PanoViewLOD",PAN_UNLOAD,NULL);
}

void PointsObject::setTransitionTimes(float moveTime, float fadeTime)
{
    _transitionTime = moveTime;
    _totalFadeTime = fadeTime;
}

void PointsObject::setAlpha(float alpha)
{
    if(!_alphaUni)
    {
	for(int i = 0; i < getNumChildNodes(); i++)
	{
	    _alphaUni = getChildNode(i)->getOrCreateStateSet()->getUniform("globalAlpha");
	    if(_alphaUni)
	    {
		break;
	    }
	}
    }

    if(_alphaUni)
    {
	_alphaUni->set(alpha);
    }

    if(_alphaRV)
    {
	_alphaRV->setValue(alpha);
    }

}

float PointsObject::getAlpha()
{
    if(!_alphaUni)
    {
	for(int i = 0; i < getNumChildNodes(); i++)
	{
	    _alphaUni = getChildNode(i)->getOrCreateStateSet()->getUniform("globalAlpha");
	    if(_alphaUni)
	    {
		break;
	    }
	}
    }

    if(_alphaUni)
    {
	float a;
	_alphaUni->get(a);
	return a;
    }

    return 1.0;
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

	osg::Vec3 movement = (_endCenter - _startCenter);
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
		startFade();
		//_storedNodeMask = _root->getNodeMask();
		//_root->setNodeMask(0);
	    }
	    else
	    {
		_activePanMarker = NULL;
		setNavigationOn(true);
	    }
	}
    }
    else if(_fadeActive)
    {
	if(_skipFrames > 0)
	{
	    _skipFrames--;
	}
	else
	{
	    _fadeTime += PluginHelper::getLastFrameDuration();
	    if(_fadeTime > _totalFadeTime)
	    {
		_fadeTime = _totalFadeTime;
	    }

	    setAlpha(1.0f - (_fadeTime / _totalFadeTime));
	    float panAlpha = _fadeTime / _totalFadeTime;
	    panAlpha *= 1.0;
	    panAlpha = std::min(panAlpha,1.0f);
	    PluginHelper::sendMessageByName("PanoViewLOD",PAN_SET_ALPHA,(char*)&panAlpha);

	    if(_fadeTime == _totalFadeTime)
	    {
		_fadeActive = false;
		detachFromScene();
		
	    }
	}
    }
    else if(_fadeInActive)
    {
	_fadeTime += PluginHelper::getLastFrameDuration();
	if(_fadeTime > _totalFadeTime)
	{
	    _fadeTime = _totalFadeTime;
	}

	float pointAlpha = _fadeTime / _totalFadeTime;
	pointAlpha *= 1.0;
	pointAlpha = std::min(pointAlpha,1.0f);

	setAlpha(pointAlpha);
	float panAlpha = 1.0f - (_fadeTime / _totalFadeTime);
	PluginHelper::sendMessageByName("PanoViewLOD",PAN_SET_ALPHA,(char*)&panAlpha);

	if(_fadeTime == _totalFadeTime)
	{
	    _fadeInActive = false;
	    _activePanMarker->panUnloaded();
	    _activePanMarker = NULL;
	    setNavigationOn(true);

	    for(int i = 0; i < getNumChildObjects(); i++)
	    {
		PanMarkerObject * pmo = dynamic_cast<PanMarkerObject*>(getChildObject(i));
		if(pmo)
		{
		    pmo->unhide();
		}
	    }
	    PluginHelper::sendMessageByName("PanoViewLOD",PAN_UNLOAD,NULL);
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

void PointsObject::menuCallback(MenuItem * item)
{
    if(_alphaRV && item == _alphaRV)
    {
	if(!_alphaUni)
	{
	    for(int i = 0; i < getNumChildNodes(); i++)
	    {
		_alphaUni = getChildNode(i)->getOrCreateStateSet()->getUniform("globalAlpha");
		if(_alphaUni)
		{
		    break;
		}
	    }
	}

	if(_alphaUni)
	{
	    _alphaUni->set(_alphaRV->getValue());
	}
    }

    SceneObject::menuCallback(item);
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

    for(int i = 0; i < getNumChildObjects(); i++)
    {
	PanMarkerObject * pmo = dynamic_cast<PanMarkerObject*>(getChildObject(i));
	if(pmo)
	{
	    pmo->hide();
	}
    }

}

void PointsObject::startFade()
{
    _fadeActive = true;

    setAlpha(1.0);

    float panAlpha = 0.0;
    PluginHelper::sendMessageByName("PanoViewLOD",PAN_SET_ALPHA,(char*)&panAlpha);

    _fadeTime = 0;
    _skipFrames = 3;
}
