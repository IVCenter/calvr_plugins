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

    FlowVis::deleteRenderer(_renderer);
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

    switch(_lastType)
    {
	case FVT_PLANE:
	{
	    osg::Vec3 point(0,1500,0), normal, origin;
	    point = point * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    origin = origin * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    normal = origin - point;
	    normal.normalize();

	    UniData pUni, nUni;
	    _renderer->getUniData("planePoint",pUni);
	    _renderer->getUniData("planeNormal",nUni);

	    ((float*)pUni.data)[0] = point.x();
	    ((float*)pUni.data)[1] = point.y();
	    ((float*)pUni.data)[2] = point.z();

	    ((float*)nUni.data)[0] = normal.x();
	    ((float*)nUni.data)[1] = normal.y();
	    ((float*)nUni.data)[2] = normal.z();
	    break;
	}
	case FVT_PLANE_VEC:
	{
	    osg::Vec3 point(0,1500,0), normal, origin, right(1,1500,0), up(0,1500,1);
	    point = point * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    origin = origin * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    right = right * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    up = up * PluginHelper::getHandMat(0) * getWorldToObjectMatrix();
	    normal = origin - point;
	    normal.normalize();

	    right = right - point;
	    right.normalize();
	    right = right * _planeVecSpacingRV->getValue();

	    up = up - point;
	    up.normalize();
	    up = up * _planeVecSpacingRV->getValue();

	    osg::Matrixf matf;
	    matf(0,0) = up.x();
	    matf(0,1) = up.y();
	    matf(0,2) = up.z();
	    matf(0,3) = 0;
	    matf(1,0) = right.x();
	    matf(1,1) = right.y();
	    matf(1,2) = right.z();
	    matf(1,3) = 0;
	    matf(2,0) = 0;
	    matf(2,1) = 0;
	    matf(2,2) = 1;
	    matf(2,3) = 0;
	    matf(3,0) = 0;
	    matf(3,1) = 0;
	    matf(3,2) = 0;
	    matf(3,3) = 1;

	    matf = osg::Matrixf::inverse(matf);

	    osg::Matrix3 m;
	    for(int i = 0; i < 3; ++i)
	    {
		for(int j = 0; j < 3; ++j)
		{
		    m(i,j) = matf(i,j);
		}
	    }

	    UniData ud;
	    _renderer->getUniData("planePoint",ud);
	    ((float*)ud.data)[0] = point.x();
	    ((float*)ud.data)[1] = point.y();
	    ((float*)ud.data)[2] = point.z();

	    _renderer->getUniData("planeNormal",ud);
	    ((float*)ud.data)[0] = normal.x();
	    ((float*)ud.data)[1] = normal.y();
	    ((float*)ud.data)[2] = normal.z();
	    
	    _renderer->getUniData("planeUp",ud);
	    ((float*)ud.data)[0] = up.x();
	    ((float*)ud.data)[1] = up.y();
	    ((float*)ud.data)[2] = up.z();

	    _renderer->getUniData("planeRight",ud);
	    ((float*)ud.data)[0] = right.x();
	    ((float*)ud.data)[1] = right.y();
	    ((float*)ud.data)[2] = right.z();

	    _renderer->getUniData("planeBasisInv",ud);
	    memcpy(ud.data,m.ptr(),9*sizeof(float));

	    //_planePointUni->set(point);
	    //_planeNormalUni->set(normal);
	    //_planeUpUni->set(up);
	    //_planeRightUni->set(right);
	    //_planeBasisInvUni->set(m);
	    break;
	}
	default:
	    break;
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
	    case FVT_PLANE:
	    {
		removeMenuItem(_alphaRV);
		break;
	    }
	    case FVT_PLANE_VEC:
	    {
		removeMenuItem(_planeVecSpacingRV);
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
	    case FVT_PLANE:
	    {
		addMenuItem(_alphaRV);

		UniData aUni;

		_renderer->getUniData("alpha",aUni);
		*((float*)aUni.data) = _alphaRV->getValue();

		_lastAttribute = "None";
		menuCallback(_loadedAttribList);
		break;
	    }
	    case FVT_PLANE_VEC:
	    {
		addMenuItem(_planeVecSpacingRV);

		_lastAttribute = "None";
		menuCallback(_loadedAttribList);
		break;
	    }
	    case FVT_VORTEX_CORES:
	    {
		std::map<std::string,std::pair<float,float> >::iterator it = _set->attribRanges.find("Vortex Cores");
		if(it != _set->attribRanges.end())
		{
		    UniData vmin,vmax;
		    _renderer->getUniData("vmin",vmin);
		    _renderer->getUniData("vmax",vmax);

		    *((float*)vmin.data) = it->second.first;
		    *((float*)vmax.data) = it->second.second;
		}

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
		case FVT_PLANE:
		case FVT_PLANE_VEC:
		case FVT_VORTEX_CORES:
		case FVT_SEP_ATT_LINES:
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

    if(item == _alphaRV)
    {
	UniData aUni;
	_renderer->getUniData("alpha",aUni);
	*((float*)aUni.data) = _alphaRV->getValue();
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
