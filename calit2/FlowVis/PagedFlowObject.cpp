#include "PagedFlowObject.h"

#include <cvrKernel/PluginHelper.h>

#include <iostream>

using namespace cvr;

PagedFlowObject::PagedFlowObject(PagedDataSet * set, osg::BoundingBox bb, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,true,showBounds)
{
    _set = set;
    _currentFrame = 0;
    _animationTime = 0.0;

    _renderer = new FlowPagedRenderer(set,0,FVT_NONE,"None");

    setBoundsCalcMode(SceneObject::MANUAL);
    setBoundingBox(bb);

    _callbackDrawable = new CallbackDrawable(_renderer,bb);
    osg::Geode * geode = new osg::Geode();
    geode->addDrawable(_callbackDrawable);
    addChild(geode);

    _animateCB = new MenuCheckbox("Animate",false);
    _animateCB->setCallback(this);
    addMenuItem(_animateCB);

    _targetFPSRV = new MenuRangeValueCompact("Target FPS",1.0,60.0,10.0);
    _targetFPSRV->setCallback(this);
    addMenuItem(_targetFPSRV);

    std::vector<std::string> visTypes;
    visTypes.push_back("None");
    visTypes.push_back("Iso Surface");
    visTypes.push_back("Plane");
    visTypes.push_back("Vector Plane");
    visTypes.push_back("Vortex Cores");
    visTypes.push_back("Sep Att Lines");
    visTypes.push_back("Volume Cuda");

    _typeList = new MenuList();
    _typeList->setCallback(this);
    _typeList->setScrollingHint(MenuList::ONE_TO_ONE);
    _typeList->setCallbackType(MenuList::ON_RELEASE);
    _typeList->setValues(visTypes);
    addMenuItem(_typeList);

    std::vector<std::string> attribList;
    attribList.push_back("None");
    for(int i = 0; i < _set->frameList[0]->pointData.size(); ++i)
    {
	attribList.push_back(_set->frameList[0]->pointData[i]->name);
    }

    _loadedAttribList = new MenuList();
    _loadedAttribList->setCallback(this);
    _loadedAttribList->setScrollingHint(MenuList::ONE_TO_ONE);
    _loadedAttribList->setCallbackType(MenuList::ON_RELEASE);
    _loadedAttribList->setValues(attribList);
    addMenuItem(_loadedAttribList);

    _lastType = FVT_NONE;
    _lastAttribute = "None";

    _isoMaxRV = NULL;

    _alphaRV = new MenuRangeValueCompact("Alpha",0.0,1.0,0.8);
    _alphaRV->setCallback(this);

    _planeVecSpacingRV = new MenuRangeValue("Spacing",0.05,1.0,0.1);
    _planeVecSpacingRV->setCallback(this);

    CVRViewer::instance()->addPerContextFrameStartCallback(this);
    CVRViewer::instance()->addPerContextPreDrawCallback(this);

    osg::Matrix scale;
    scale.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
    setTransform(scale);
}

PagedFlowObject::~PagedFlowObject()
{
    CVRViewer::instance()->removePerContextFrameStartCallback(this);
    CVRViewer::instance()->removePerContextPreDrawCallback(this);
}

void PagedFlowObject::preFrame()
{
    if(_animateCB->getValue() && _set && _set->frameList.size())
    {
	_animationTime += PluginHelper::getLastFrameDuration();
	if(_animationTime > 1.0 / _targetFPSRV->getValue() && _renderer->advance())
	{
	    _currentFrame = (_currentFrame + 1) % _set->frameList.size();
	    int next = (_currentFrame + 1) % _set->frameList.size();
	    _renderer->setNextFrame(next);
	    _animationTime = 0.0;
	}
    }

    _renderer->preFrame();
}

void PagedFlowObject::postFrame()
{
    _renderer->postFrame();
}

void PagedFlowObject::menuCallback(cvr::MenuItem * item)
{
    if(item == _typeList)
    {
	if(_typeList->getIndex() == _lastType)
	{
	    return;
	}

	// cleanup last type
	switch(_lastType)
	{
	    case FVT_ISO_SURFACE:
	    {
		if(_isoMaxRV)
		{
		    removeMenuItem(_isoMaxRV);
		    delete _isoMaxRV;
		    _isoMaxRV = NULL;
		}
		break;
	    }
	    default:
		break;
	}

	_lastType = (FlowVisType)_typeList->getIndex();

	// setup new type
	switch(_lastType)
	{
	    case FVT_ISO_SURFACE:
	    {
		// setup iso max menu item
		_lastAttribute = "None";
		menuCallback(_loadedAttribList);
		break;
	    }
	    default:
		break;
	}

	_renderer->setType((FlowVisType)_typeList->getIndex(),_loadedAttribList->getValue());
    }

    if(item == _loadedAttribList)
    {
	//std::cerr << "Setting attribute: " << _loadedAttribList->getValue() << std::endl;
	if(_loadedAttribList->getValue() == _lastAttribute)
	{
	    return;
	}

	_lastAttribute = _loadedAttribList->getValue();

	PagedDataAttrib * attrib = NULL;
	for(int i = 0; i < _set->frameList[_currentFrame]->pointData.size(); ++i)
	{
	    if(_set->frameList[_currentFrame]->pointData[i]->name == _lastAttribute)
	    {
		attrib = _set->frameList[_currentFrame]->pointData[i];
		break;
	    }
	}

	if(_isoMaxRV)
	{
	    removeMenuItem(_isoMaxRV);
	    delete _isoMaxRV;
	    _isoMaxRV = NULL;
	}

	if(attrib && _set->attribRanges.find(_lastAttribute) != _set->attribRanges.end())
	{
	    //std::cerr << "Found attrib" << std::endl;
	    switch(_lastType)
	    {
		case FVT_NONE:
		{
		    if(_lastAttribute != "None")
		    {
			UniData min, max;
			if(attrib->dataType == VDT_INT)
			{
			    _renderer->getUniData("mini",min);
			    _renderer->getUniData("maxi",max);
			    *((int*)min.data) = (int)_set->attribRanges[_lastAttribute].first;
			    *((int*)max.data) = (int)_set->attribRanges[_lastAttribute].second;
			}
			else
			{
			    _renderer->getUniData("minf",min);
			    _renderer->getUniData("maxf",max);
			    *((float*)min.data) = _set->attribRanges[_lastAttribute].first;
			    *((float*)max.data) = _set->attribRanges[_lastAttribute].second;
			}
		    }
		    break;
		}
		case FVT_ISO_SURFACE:
		{
		    _isoMaxRV = new MenuRangeValue("ISO Value",_set->attribRanges[_lastAttribute].first,_set->attribRanges[_lastAttribute].second,_set->attribRanges[_lastAttribute].second);
		    _isoMaxRV->setCallback(this);
		    addMenuItem(_isoMaxRV);

		    UniData max;
		    _renderer->getUniData("isoMax",max);
		    *((float*)max.data) = _set->attribRanges[_lastAttribute].second;

		    break;
		}
		default:
		    break;
	    }
	}

	_renderer->setType((FlowVisType)_typeList->getIndex(),_loadedAttribList->getValue());
    }

    if(item == _isoMaxRV)
    {
	UniData max;
	_renderer->getUniData("isoMax",max);
	*((float*)max.data) = _isoMaxRV->getValue();
    }

    if(item == _animateCB)
    {
	if(_animateCB->getValue())
	{
	    int next = (_currentFrame + 1) % _set->frameList.size();
	    _renderer->setNextFrame(next);
	}
	else
	{
	    _renderer->setNextFrame(_currentFrame);
	}
    }
}

void PagedFlowObject::perContextCallback(int contextid, PerContextCallback::PCCType type) const
{
    if(type == PCC_FRAME_START)
    {
	_renderer->frameStart(contextid);
    }
    else if(type == PCC_PRE_DRAW)
    {
	_renderer->preDraw(contextid);
    }
}
