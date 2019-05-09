#include "LinearRegFunc.h"

#include <cvrKernel/ComController.h>

#include <octave/config.h>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h>

using namespace cvr;

LinearRegFunc::LinearRegFunc()
{
    _lrGeometry = new osg::Geometry();
    _lrGeometry->setUseDisplayList(false);
    _lrGeometry->setUseVertexBufferObjects(true);
    osg::Vec3Array * verts = new osg::Vec3Array(2);
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(1,1,0,1);
    _lrGeometry->setVertexArray(verts);
    _lrGeometry->setColorArray(colors);
    _lrGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _lrGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,2));
    
    _lrLineWidth = new osg::LineWidth();

    _lrGeometry->getOrCreateStateSet()->setAttributeAndModes(_lrLineWidth,osg::StateAttribute::ON);
    _lrGeometry->getOrCreateStateSet()->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    _lrBoundsCallback = new SetBoundsCallback;

    _lrGeometry->setComputeBoundingBoxCallback(_lrBoundsCallback.get());

    _healthyIntersectTime = 0;
}

LinearRegFunc::~LinearRegFunc()
{
}

void LinearRegFunc::added(osg::Geode * geode)
{
    geode->addDrawable(_lrGeometry);
}

void LinearRegFunc::removed(osg::Geode * geode)
{
    geode->removeDrawable(_lrGeometry);
}

void LinearRegFunc::update(float width, float height, std::map<std::string, GraphDataInfo> & data, std::map<std::string, std::pair<float,float> > & displayRanges, std::map<std::string,std::pair<int,int> > & dataPointRanges)
{
    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_lrGeometry->getVertexArray());
    if(!verts)
    {
	return;
    }

    if(!dataPointRanges.size() || dataPointRanges.begin()->second.first < 0 || dataPointRanges.begin()->second.second < 0 || dataPointRanges.begin()->second.first > dataPointRanges.begin()->second.second)
    {
	verts->at(0) = osg::Vec3(0,0,0);
	verts->at(1) = osg::Vec3(0,0,0);
	verts->dirty();
	return;
    }

    std::map<std::string,std::pair<int,int> >::iterator pit = dataPointRanges.begin();
    std::map<std::string, GraphDataInfo>::iterator dit = data.find(pit->first);
    if(dit == data.end())
    {
	verts->at(0) = osg::Vec3(0,0,0);
	verts->at(1) = osg::Vec3(0,0,0);
	verts->dirty();
	return;
    }


    if((pit->second.second - pit->second.first) < 1)
    {
	verts->at(0) = osg::Vec3(0,0,0);
	verts->at(1) = osg::Vec3(0,0,0);
	verts->dirty();
	return;
    }

    Matrix XMat((pit->second.second - pit->second.first)+1,2);
    Matrix ZMat((pit->second.second - pit->second.first)+1,1);

    for(int i = pit->second.first; i <= pit->second.second; ++i)
    {
	int matindex = i - pit->second.first;
	XMat(matindex,0) = 1.0;
	XMat(matindex,1) = dit->second.data->at(i).x();
	ZMat(matindex) = dit->second.data->at(i).z();
    }

    Matrix XTMat = XMat.transpose();
    Matrix resMat = (XTMat*XMat).pseudo_inverse()*XTMat*ZMat;
    
    std::map<std::string,std::pair<float,float> >::iterator hit = _healthyRangeMap.begin();
    std::map<std::string,std::pair<float,float> >::iterator dataIt = _dataRangeMap.begin();
    std::map<std::string,std::pair<time_t,time_t> >::iterator timeIt = _timeRangeMap.begin();
    if(hit != _healthyRangeMap.end() && dataIt != _dataRangeMap.end() && timeIt != _timeRangeMap.end())
    {
	time_t intersect = 0;
	if(hit->second.first != FLT_MIN && hit->second.first != 0.0)
	{
	    float minNorm = (hit->second.first - dataIt->second.first) / (dataIt->second.second - dataIt->second.first);
	    float x = (minNorm - resMat(0)) / resMat(1);
	    if(x > 1.0)
	    {
		intersect = timeIt->second.first + (time_t)(x * ((float)(timeIt->second.second - timeIt->second.first)));
	    }
	}

	if(hit->second.second != FLT_MAX && hit->second.second != 0.0)
	{
	    float minNorm = (hit->second.second - dataIt->second.first) / (dataIt->second.second - dataIt->second.first);
	    float x = (minNorm - resMat(0)) / resMat(1);
	    if(x > 1.0)
	    {
		time_t mintersect = timeIt->second.first + (time_t)(x * ((float)(timeIt->second.second - timeIt->second.first)));
		if(intersect == 0 || mintersect < intersect)
		{
		    intersect = mintersect;
		}
	    }
	}
	_healthyIntersectTime = intersect;
    }

    float xPosMin, xPosMax;
    float zPosMin, zPosMax;
    
     std::map<std::string, std::pair<float,float> >::iterator displayIt = displayRanges.begin();

    if(resMat(1) == 0.0)
    {
	xPosMin = 0.0;
	xPosMax = 1.0;
	zPosMin = zPosMax = resMat(0);
    }
    else
    {
	float y = displayIt->second.first * resMat(1) + resMat(0);
	if(y >= 0.0 && y <= 1.0)
	{
	    xPosMin = 0.0;
	    zPosMin = y;
	}
	else if(resMat(1) > 0.0)
	{
	    float x = -resMat(0) / resMat(1);
	    xPosMin = (x - displayIt->second.first) / (displayIt->second.second - displayIt->second.first);
	    zPosMin = 0.0;
	}
	else
	{
	    float x = (1.0 - resMat(0)) / resMat(1);
	    xPosMin = (x - displayIt->second.first) / (displayIt->second.second - displayIt->second.first);
	    zPosMin = 1.0;
	}

	y = displayIt->second.second * resMat(1) + resMat(0);
	if(y >= 0.0 && y <= 1.0)
	{
	    xPosMax = 1.0;
	    zPosMax = y;
	}
	else if(resMat(1) > 0.0)
	{
	    float x = (1.0-resMat(0)) / resMat(1);
	    xPosMax = (x - displayIt->second.first) / (displayIt->second.second - displayIt->second.first);
	    zPosMax = 1.0;
	}
	else
	{
	    float x = -resMat(0) / resMat(1);
	    xPosMax = (x - displayIt->second.first) / (displayIt->second.second - displayIt->second.first);
	    zPosMax = 0.0;
	}
    }


    xPosMin = (xPosMin * width) - (width / 2.0);
    xPosMax = (xPosMax * width) - (width / 2.0);
    zPosMin = (zPosMin * height) - (height / 2.0);
    zPosMax = (zPosMax * height) - (height / 2.0);

    verts->at(0) = osg::Vec3(xPosMin,-0.5,zPosMin);
    verts->at(1) = osg::Vec3(xPosMax,-0.5,zPosMax);
    verts->dirty();

    float avglen = (width + height) / 2.0;
    _lrLineWidth->setWidth(avglen * 0.05 * GraphGlobals::getPointLineScale() * GraphGlobals::getPointLineScale());
    if(ComController::instance()->isMaster())
    {
	_lrLineWidth->setWidth(_lrLineWidth->getWidth() * GraphGlobals::getMasterLineScale());
    }

    _lrBoundsCallback->bbox.set(-width/2.0,-3,-height/2.0,width/2.0,1,height/2.0);
    _lrGeometry->dirtyBound();
    _lrGeometry->getBoundingBox();

}

void LinearRegFunc::setDataRange(std::string name, float min, float max)
{
    _dataRangeMap[name] = std::pair<float,float>(min,max);
}

void LinearRegFunc::setTimeRange(std::string name, time_t min, time_t max)
{
    _timeRangeMap[name] = std::pair<time_t,time_t>(min,max);
}

void LinearRegFunc::setHealthyRange(std::string name, float min, float max)
{
    _healthyRangeMap[name] = std::pair<float,float>(min,max);
}
