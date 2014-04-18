#include "PagedFlowObject.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ScreenConfig.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/OsgMath.h>

#include <iostream>

#include <sys/time.h>

using namespace cvr;

PagedFlowObject::PagedFlowObject(PagedDataSet * set, osg::BoundingBox bb, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,true,showBounds)
{
    _set = set;
    _currentFrame = 0;
    _animationTime = 0.0;

    initCudaInfo();
    initContextRenderCount();

    // get size of vbo cache in KB, 1GB default
    int cacheSize = ConfigManager::getInt("value","Plugin.FlowVis.VBOCacheSize",1048576);
    _renderer = new FlowPagedRenderer(set,0,FVT_NONE,"None",cacheSize);

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
    visTypes.push_back("LIC Cuda");

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
#ifdef PRINT_TIMING
    struct timeval start,end;
    gettimeofday(&start,NULL);
#endif

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

	    UniData ud;

	    right = right - point;
	    right.normalize();

	    _renderer->getUniData("planeRightNorm",ud);
	    ((float*)ud.data)[0] = right.x();
	    ((float*)ud.data)[1] = right.y();
	    ((float*)ud.data)[2] = right.z();

	    right = right * _planeVecSpacingRV->getValue();

	    up = up - point;
	    up.normalize();

	    _renderer->getUniData("planeUpNorm",ud);
	    ((float*)ud.data)[0] = up.x();
	    ((float*)ud.data)[1] = up.y();
	    ((float*)ud.data)[2] = up.z();

	    up = up * _planeVecSpacingRV->getValue();

	    _renderer->getUniData("planeBasisLength",ud);
	    ((float*)ud.data)[0] = _planeVecSpacingRV->getValue();

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

	    break;
	}
	case FVT_LIC_CUDA:
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

	    up = up - point;
	    up.normalize();

	    std::vector<osg::Vec3> edgeIntersectionList;

	    getBoundsPlaneIntersectPoints(point,normal,_set->bb,edgeIntersectionList);

	    std::vector<osg::Vec3> viewportIntersectionList;
	    getPlaneViewportIntersection(point,normal,viewportIntersectionList);

	    float basisXMin = FLT_MAX;
	    float basisXMax = -FLT_MAX;
	    float basisYMin = FLT_MAX;
	    float basisYMax = -FLT_MAX;

	    for(int i = 0; i < edgeIntersectionList.size(); ++i)
	    {
		osg::Vec3 tempP = edgeIntersectionList[i]-point;
		
		// projection method
		osg::Vec3 basisPoint;
		basisPoint.x() = tempP * up;
		basisPoint.y() = tempP * right;

		//std::cerr << "Basis Point x: " << basisPoint.x() << " y: " << basisPoint.y() << " z: " << basisPoint.z() << std::endl;
		basisXMin = std::min(basisXMin,basisPoint.x());
		basisXMax = std::max(basisXMax,basisPoint.x());
		basisYMin = std::min(basisYMin,basisPoint.y());
		basisYMax = std::max(basisYMax,basisPoint.y());
	    }

	    // TODO: handle case of first call
	    if(!edgeIntersectionList.size())
	    {
		break;
	    }

	    // only use viewport bounds if all corner rays hit the lic plane
	    // also don't do this for clusters, since they have different viewports per node
	    if(viewportIntersectionList.size() == 4 && !ComController::instance()->getNumSlaves())
	    {
		//std::cerr << "Using viewport bounds" << std::endl;

		// factor in viewport
		float vpBasisXMin = FLT_MAX;
		float vpBasisXMax = -FLT_MAX;
		float vpBasisYMin = FLT_MAX;
		float vpBasisYMax = -FLT_MAX;

		for(int i = 0; i < viewportIntersectionList.size(); ++i)
		{
		    osg::Vec3 tempP = viewportIntersectionList[i]-point;

		    osg::Vec3 basisPoint;
		    basisPoint.x() = tempP * up;
		    basisPoint.y() = tempP * right;

		    vpBasisXMin = std::min(vpBasisXMin,basisPoint.x());
		    vpBasisXMax = std::max(vpBasisXMax,basisPoint.x());
		    vpBasisYMin = std::min(vpBasisYMin,basisPoint.y());
		    vpBasisYMax = std::max(vpBasisYMax,basisPoint.y());
		}

		float vpBasisXPadding = (vpBasisXMax - vpBasisXMin)*0.20;
		float vpBasisYPadding = (vpBasisYMax - vpBasisYMin)*0.20;
		vpBasisXMin -= vpBasisXPadding;
		vpBasisXMax += vpBasisXPadding;
		vpBasisYMin -= vpBasisYPadding;
		vpBasisYMax += vpBasisYPadding;

		//std::cerr << "Old - BasisXMin: " << basisXMin << " BasisXMax: " << basisXMax << std::endl;
		//std::cerr << "Old - BasisYMin: " << basisYMin << " BasisYMax: " << basisYMax << std::endl;

		if(vpBasisXMin > basisXMin)
		{
		    basisXMin = vpBasisXMin;
		}
		if(vpBasisYMin > basisYMin)
		{
		    basisYMin = vpBasisYMin;
		}
		if(vpBasisXMax < basisXMax)
		{
		    basisXMax = vpBasisXMax;
		}
		if(vpBasisYMax < basisYMax)
		{
		    basisYMax = vpBasisYMax;
		}
	    }

	    if(basisXMin >= basisXMax || basisYMin >= basisYMax)
	    {
		break;
	    }

	    float textureSize = LIC_TEXTURE_SIZE;

	    float basisXRange = basisXMax - basisXMin;
	    float basisYRange = basisYMax - basisYMin;
	    float basisMaxRange = std::max(basisXRange,basisYRange);

	    basisXMin = (basisXMin + (basisXRange / 2.0)) - (basisMaxRange / 2.0);
	    basisXMax = (basisXMin + (basisXRange / 2.0)) + (basisMaxRange / 2.0);
	    basisYMin = (basisYMin + (basisYRange / 2.0)) - (basisMaxRange / 2.0);
	    basisYMax = (basisYMin + (basisYRange / 2.0)) + (basisMaxRange / 2.0);

	    osg::Vec3 basisCenter = up * (basisXMin + (basisXRange/2.0)) + right * (basisYMin + (basisYRange/2.0));
	    basisCenter += point;

	    float basisScale = basisMaxRange / textureSize;
	   
	    float centerOffsetX = ((point - basisCenter) * up) / basisScale;
	    float centerOffsetY = ((point - basisCenter) * right) / basisScale;

	    UniData ud;
	    
	    _renderer->getUniData("planeUpNorm",ud);
	    ((float*)ud.data)[0] = up.x();
	    ((float*)ud.data)[1] = up.y();
	    ((float*)ud.data)[2] = up.z();

	    _renderer->getUniData("planeRightNorm",ud);
	    ((float*)ud.data)[0] = right.x();
	    ((float*)ud.data)[1] = right.y();
	    ((float*)ud.data)[2] = right.z();
	    
	    right = right * basisScale;
	    up = up * basisScale;

	    _renderer->getUniData("planeBasisLength",ud);
	    ((float*)ud.data)[0] = basisScale;

	    _renderer->getUniData("planeBasisXMin",ud);
	    ((float*)ud.data)[0] = (basisXMin / basisScale) + centerOffsetX;
	    _renderer->getUniData("planeBasisXMax",ud);
	    ((float*)ud.data)[0] = (basisXMax / basisScale) + centerOffsetX;
	    _renderer->getUniData("planeBasisYMin",ud);
	    ((float*)ud.data)[0] = (basisYMin / basisScale) + centerOffsetY;
	    _renderer->getUniData("planeBasisYMax",ud);
	    ((float*)ud.data)[0] = (basisYMax / basisScale) + centerOffsetY;

	    _renderer->getUniData("planePoint",ud);
	    ((float*)ud.data)[0] = basisCenter.x();
	    ((float*)ud.data)[1] = basisCenter.y();
	    ((float*)ud.data)[2] = basisCenter.z();

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

	    break;
	}
	default:
	    break;
    }

#ifdef PRINT_TIMING
    gettimeofday(&end,NULL);
    std::cerr << "PagedFlowObject time (without renderer): " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec)/1000000.0) << std::endl;
#endif

    _renderer->preFrame();
}

void PagedFlowObject::postFrame()
{
    switch(_lastType)
    {
	case FVT_LIC_CUDA:
	{
	    // check again for frame advance before the next LIC calculation starts
	    if(_animateCB->getValue() && _set && _set->frameList.size())
	    {
		if(_animationTime > 1.0 / _targetFPSRV->getValue() && _renderer->advance())
		{
		    _currentFrame = (_currentFrame + 1) % _set->frameList.size();
		    int next = (_currentFrame + 1) % _set->frameList.size();
		    _renderer->setNextFrame(next);
		    _animationTime = 0.0;
		}
	    }
	}
	default:
	    break;
    }

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
	    case FVT_LIC_CUDA:
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
	    case FVT_LIC_CUDA:
	    {
		addMenuItem(_alphaRV);

		UniData aUni;

		_renderer->getUniData("alpha",aUni);
		*((float*)aUni.data) = _alphaRV->getValue();

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

void PagedFlowObject::getBoundsPlaneIntersectPoints(osg::Vec3 point, osg::Vec3 normal, osg::BoundingBox & bounds, std::vector<osg::Vec3> & intersectList)
{
    // check intersection of plane with all edges of aabb

    osg::Vec3 bpoint1, bpoint2;
    osg::Vec3 intersect;
    float w;

    bpoint1 = osg::Vec3(bounds.xMin(),bounds.yMin(),bounds.zMax());
    bpoint2 = osg::Vec3(bounds.xMin(),bounds.yMax(),bounds.zMax());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMin(),bounds.yMax(),bounds.zMax());
    bpoint2 = osg::Vec3(bounds.xMax(),bounds.yMax(),bounds.zMax());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMax(),bounds.yMin(),bounds.zMax());
    bpoint2 = osg::Vec3(bounds.xMax(),bounds.yMax(),bounds.zMax());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMin(),bounds.yMin(),bounds.zMax());
    bpoint2 = osg::Vec3(bounds.xMax(),bounds.yMin(),bounds.zMax());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMin(),bounds.yMax(),bounds.zMin());
    bpoint2 = osg::Vec3(bounds.xMin(),bounds.yMax(),bounds.zMax());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMin(),bounds.yMin(),bounds.zMin());
    bpoint2 = osg::Vec3(bounds.xMin(),bounds.yMin(),bounds.zMax());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMax(),bounds.yMin(),bounds.zMin());
    bpoint2 = osg::Vec3(bounds.xMax(),bounds.yMin(),bounds.zMax());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMax(),bounds.yMax(),bounds.zMin());
    bpoint2 = osg::Vec3(bounds.xMax(),bounds.yMax(),bounds.zMax());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMin(),bounds.yMin(),bounds.zMin());
    bpoint2 = osg::Vec3(bounds.xMin(),bounds.yMax(),bounds.zMin());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMin(),bounds.yMax(),bounds.zMin());
    bpoint2 = osg::Vec3(bounds.xMax(),bounds.yMax(),bounds.zMin());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMax(),bounds.yMin(),bounds.zMin());
    bpoint2 = osg::Vec3(bounds.xMax(),bounds.yMax(),bounds.zMin());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);

    bpoint1 = osg::Vec3(bounds.xMin(),bounds.yMin(),bounds.zMin());
    bpoint2 = osg::Vec3(bounds.xMax(),bounds.yMin(),bounds.zMin());
    checkAndAddIntersect(bpoint1,bpoint2,point,normal,intersectList);
}

void PagedFlowObject::checkAndAddIntersect(osg::Vec3 & p1,osg::Vec3 & p2,osg::Vec3 & planep, osg::Vec3 & planen,std::vector<osg::Vec3> & intersectList)
{
    osg::Vec3 intersect;
    float w;

    //std::cerr << "Point1: " << p1.x() << " " << p1.y() << " " << p1.z() << std::endl;
    //std::cerr << "Point2: " << p2.x() << " " << p2.y() << " " << p2.z() << std::endl;

    if(linePlaneIntersectionRef(p1,p2,planep,planen,intersect,w))
    {
	w = w / (p2-p1).length();
	//std::cerr << "w: " << w << std::endl;
	// see if hit is on the bounds edge
	if(w >= 0.0 && w <= 1.0)
	{
	    intersectList.push_back(intersect);
	}
    }
}

void PagedFlowObject::initCudaInfo()
{
    std::map<int,std::pair<int,int> > initInfo;

    for(int i = 0; i < ScreenConfig::instance()->getNumWindows(); ++i)
    {
	initInfo[i].first = ScreenConfig::instance()->getCudaDevice(i);
	initInfo[i].second = ScreenConfig::instance()->getNumContexts(i);
    }

    FlowPagedRenderer::setCudaInitInfo(initInfo);
}

void PagedFlowObject::initContextRenderCount()
{
    std::map<int,int> drawCounts;

    // TODO: determine this based on number of channels and stereo modes
    // for now assume one draw per context
    for(int i = 0; i < ScreenConfig::instance()->getNumWindows(); ++i)
    {
	drawCounts[i] = 1;
    }

    FlowPagedRenderer::setContextRenderCount(drawCounts);
}

void PagedFlowObject::getPlaneViewportIntersection(const osg::Vec3 & planePoint, const osg::Vec3 & planeNormal, std::vector<osg::Vec3> & intersectList)
{
    ScreenInfo * screen = ScreenConfig::instance()->getScreenInfo(0);

    std::vector<osg::Vec3> vpCorners;
    vpCorners.push_back(osg::Vec3(screen->width/2.0,0,screen->height/2.0));
    vpCorners.push_back(osg::Vec3(screen->width/-2.0,0,screen->height/2.0));
    vpCorners.push_back(osg::Vec3(screen->width/2.0,0,screen->height/-2.0));
    vpCorners.push_back(osg::Vec3(screen->width/-2.0,0,screen->height/-2.0));

    osg::Vec3 viewerPos = PluginHelper::getHeadMat(0).getTrans();

    // transform points to world space
    for(int i = 0; i < vpCorners.size(); ++i)
    {
	vpCorners[i] = vpCorners[i] * screen->transform;
    }

    // transform into sim space
    viewerPos = viewerPos * getWorldToObjectMatrix();
    for(int i = 0; i < vpCorners.size(); ++i)
    {
	vpCorners[i] = vpCorners[i] * getWorldToObjectMatrix();
    }

    // find plane intersection
    for(int i = 0; i < vpCorners.size(); ++i)
    {
	float w;
	osg::Vec3 intersect;

	if(linePlaneIntersectionRef(viewerPos,vpCorners[i],planePoint,planeNormal,intersect,w) && w > 0.0)
	{
	    intersectList.push_back(intersect);
	}
    }
}
