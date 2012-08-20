#include "DataGraph.h"

#include <cvrKernel/CalVR.h>
#include <cvrUtil/OsgMath.h>

#include <osgText/Text>
#include <osg/Geode>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace cvr;

DataGraph::DataGraph()
{
    _root = new osg::MatrixTransform();
    _clipNode = new osg::ClipNode();
    _graphTransform = new osg::MatrixTransform();
    _graphGeode = new osg::Geode();
    _axisGeode = new osg::Geode();
    _axisGeometry = new osg::Geometry();
    _bgGeometry = new osg::Geometry();

    _root->addChild(_axisGeode);
    _root->addChild(_graphTransform);
    _root->addChild(_clipNode);
    _graphTransform->addChild(_graphGeode);
    _graphGeode->addDrawable(_bgGeometry);
    _axisGeode->addDrawable(_axisGeometry);

    _point = new osg::Point();
    _lineWidth = new osg::LineWidth();

    osg::StateSet * stateset = getGraphRoot()->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    stateset->setAttributeAndModes(_point,osg::StateAttribute::ON);
    _clipNode->getOrCreateStateSet()->setAttributeAndModes(_lineWidth,osg::StateAttribute::ON);

    _width = _height = 1000.0;

    osg::Vec4 color(1.0,1.0,1.0,1.0);

    osg::Geometry * geo = _bgGeometry.get();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(osg::Vec3(1.0,1,1.0));
    verts->push_back(osg::Vec3(1.0,1,0));
    verts->push_back(osg::Vec3(0,1,0));
    verts->push_back(osg::Vec3(0,1,1.0));

    geo->setVertexArray(verts);

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(
	    osg::PrimitiveSet::QUADS,0);

    ele->push_back(0);
    ele->push_back(1);
    ele->push_back(2);
    ele->push_back(3);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(color);

    osg::TemplateIndexArray<unsigned int,osg::Array::UIntArrayType,4,4> *colorIndexArray;
    colorIndexArray = new osg::TemplateIndexArray<unsigned int,
		    osg::Array::UIntArrayType,4,4>;
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);

    geo->setColorArray(colors);
    geo->setColorIndices(colorIndexArray);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    _xAxisTimestamp = false;
    _minDisplayX = 0;
    _maxDisplayX = 1.0;
    _minDisplayZ = 0;
    _maxDisplayZ = 1.0;

    //_clipNode->addClipPlane(new osg::ClipPlane(0));
    //_clipNode->addClipPlane(new osg::ClipPlane(1));
    //_clipNode->addClipPlane(new osg::ClipPlane(2));
    //_clipNode->addClipPlane(new osg::ClipPlane(3));

    _font = osgText::readFontFile(CalVR::instance()->getHomeDir() + "/resources/arial.ttf");

    makeHover();
}

DataGraph::~DataGraph()
{
}

void DataGraph::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;
    update();
}

void DataGraph::addGraph(std::string name, osg::Vec3Array * points, GraphDisplayType displayType, std::string xLabel, std::string zLabel, osg::Vec4 color, osg::Vec4Array * perPointColor, osg::Vec4Array * secondaryPerPointColor)
{
    if(_dataInfoMap.find(name) != _dataInfoMap.end())
    {
	std::cerr << "Error: Graph " << name << " has already been added to DataGraph." << std::endl;
	return;
    }

    //std::cerr << "Points: " << points->size() << std::endl;

    GraphDataInfo gdi;
    gdi.name = name;
    gdi.data = points;
    gdi.colorArray = perPointColor;
    gdi.secondaryColorArray = secondaryPerPointColor;
    gdi.color = color;
    gdi.displayType = displayType;
    gdi.xLabel = xLabel;
    gdi.zLabel = zLabel;
    gdi.xAxisType = LINEAR;
    gdi.zAxisType = LINEAR;
    gdi.xMin = 0.0;
    gdi.xMax = 1.0;
    gdi.zMin = 0.0;
    gdi.zMax = 1.0;
    _dataInfoMap[name] = gdi;

    osg::Geometry * geometry = new osg::Geometry();
    geometry->setUseDisplayList(false);
    geometry->setUseVertexBufferObjects(true);
    geometry->setVertexArray(points);

    if(!perPointColor || perPointColor->size() != points->size())
    {
	osg::Vec4Array * colorArray = new osg::Vec4Array(1);
	colorArray->at(0) = color;
	geometry->setColorArray(colorArray);
	geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    }
    else
    {
	geometry->setColorArray(perPointColor);
	geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    for(int i = 0; i < points->size(); i++)
    {
	osg::Vec3 p = points->at(i);
	//std::cerr << "Point x: " << p.x() << " y: " << p.y() << " z: " << p.z() << std::endl;
    }
    if(perPointColor)
    {
	for(int i = 0; i < perPointColor->size(); i++)
	{
	    osg::Vec4 c = perPointColor->at(i);
	    //std::cerr << "Color r: " << c.x() << " g: " << c.y() << " b: " << c.z() << " a: " << c.w() << std::endl;
	}
    }

    geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,points->size()));
    if(displayType == POINTS_WITH_LINES && points->size())
    {
	geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP,0,points->size()));
    }

    _graphGeometryMap[name] = geometry;
    osg::Geode * geode = new osg::Geode();
    geode->addDrawable(geometry);

    _graphTransformMap[name] = new osg::MatrixTransform();
    _graphTransformMap[name]->addChild(geode);
    _clipNode->addChild(_graphTransformMap[name]);

    update();
}

void DataGraph::setXDataRangeTimestamp(std::string graphName, time_t & start, time_t & end)
{
    if(_dataInfoMap.find(graphName) != _dataInfoMap.end())
    {
	if(!_xAxisTimestamp)
	{
	    _xAxisTimestamp = true;
	    _minDisplayXT = start;
	    _maxDisplayXT = end;
	}
	_dataInfoMap[graphName].xMinT = start;
	_dataInfoMap[graphName].xMaxT = end;
	_dataInfoMap[graphName].xAxisType = TIMESTAMP;
	updateAxis();
    }
}

void DataGraph::setZDataRange(std::string graphName, float min, float max)
{
    if(_dataInfoMap.find(graphName) != _dataInfoMap.end())
    {
	_dataInfoMap[graphName].zMin = min;
	_dataInfoMap[graphName].zMax = max;
	updateAxis();
    }
}

void DataGraph::setXDisplayRange(float min, float max)
{
    _minDisplayX = min;
    _maxDisplayX = max;
    update();
}

void DataGraph::setXDisplayRangeTimestamp(time_t & start, time_t & end)
{
    _minDisplayXT = start;
    _maxDisplayXT = end;
    update();
}

void DataGraph::setZDisplayRange(float min, float max)
{
    _minDisplayZ = min;
    _maxDisplayZ = max;
    update();
}

osg::MatrixTransform * DataGraph::getGraphRoot()
{
    return _root.get();
}

void DataGraph::getGraphNameList(std::vector<std::string> & nameList)
{
    for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
    {
	nameList.push_back(it->first);
    }
}

time_t DataGraph::getMaxTimestamp(std::string graphName)
{
    if(_dataInfoMap.find(graphName) != _dataInfoMap.end())
    {
	return _dataInfoMap[graphName].xMaxT;
    }
    return 0;
}

time_t DataGraph::getMinTimestamp(std::string graphName)
{
    if(_dataInfoMap.find(graphName) != _dataInfoMap.end())
    {
	return _dataInfoMap[graphName].xMinT;
    }
    return 0;
}

bool DataGraph::displayHoverText(osg::Matrix & mat)
{
    std::string selectedGraph;
    int selectedPoint = -1;
    osg::Vec3 point;
    float dist;

    for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
    {
	if(!it->second.data->size())
	{
	    continue;
	}

	osg::Matrix invT = osg::Matrix::inverse(_graphTransformMap[it->first]->getMatrix());
	osg::Vec3 point1, point2(0,1000,0);
	
	point1 = point1 * mat * invT;
	point2 = point2 * mat * invT;
	
	osg::Vec3 planePoint, planeNormal(0,-1,0);
	float w;

	osg::Vec3 intersect;
	if(linePlaneIntersectionRef(point1,point2,planePoint,planeNormal,intersect,w))
	{
	    //std::cerr << "Intersect Point x: " << intersect.x() << " y: " << intersect.y() << " z: " << intersect.z() << std::endl;

	    if(intersect.x() < 0.0 || intersect.x() > 1.0 || intersect.z() < 0.0 || intersect.z() > 1.0)
	    {
		continue;
	    }

	    // find nearest point on x axis
	    int start, end, current;
	    start = 0;
	    end = it->second.data->size() - 1;
	    while(end - start > 1)
	    {
		current = start + ((end-start) / 2);
		if(intersect.x() < it->second.data->at(current).x())
		{
		    end = current;
		}
		else
		{
		    start = current;
		}
	    }

	    if(end == start)
	    {
		current = start;
	    }
	    else
	    {
		float startx, endx;
		startx = it->second.data->at(start).x();
		endx = it->second.data->at(end).x();

		if(fabs(intersect.x() - startx) > fabs(endx - intersect.x()))
		{
		    current = end;
		}
		else
		{
		    current = start;
		}
	    }
	    osg::Vec3 currentPoint = it->second.data->at(current);
	    currentPoint = currentPoint * _graphTransformMap[it->first]->getMatrix();

	    intersect = intersect * _graphTransformMap[it->first]->getMatrix();
	    if(selectedPoint < 0)
	    {
		selectedPoint = current;
		selectedGraph = it->first;
		point = currentPoint;
		dist = (intersect - currentPoint).length();
	    }
	    else if((intersect - currentPoint).length() < dist)
	    {
		selectedPoint = current;
		selectedGraph = it->first;
		point = currentPoint;
		dist = (intersect - currentPoint).length();
	    }
	}
	else
	{
	    break;
	}
    }

    if(selectedPoint >= 0 && dist < (_width + _height) * 0.02 / 2.0)
    {
	//std::cerr << "selecting point" << std::endl;
	if(selectedGraph != _hoverGraph || selectedPoint != _hoverPoint)
	{
	    std::stringstream textss;
	    time_t time;
	    float value;
	    value = _dataInfoMap[selectedGraph].zMin + ((_dataInfoMap[selectedGraph].zMax - _dataInfoMap[selectedGraph].zMin) * _dataInfoMap[selectedGraph].data->at(selectedPoint).z());
	    time = _dataInfoMap[selectedGraph].xMinT + (time_t)((_dataInfoMap[selectedGraph].xMaxT - _dataInfoMap[selectedGraph].xMinT) * _dataInfoMap[selectedGraph].data->at(selectedPoint).x());
	    textss << "x: " << ctime(&time) << "y: " << value << " " << _dataInfoMap[selectedGraph].zLabel;
	    _hoverText->setText(textss.str());
	    _hoverText->setCharacterSize(1.0);

	    float targetHeight = _height * 0.07;
	    osg::BoundingBox bb = _hoverText->getBound();
	    _hoverText->setCharacterSize(targetHeight / (bb.zMax() - bb.zMin()));

	    bb = _hoverText->getBound();
	    osg::Matrix bgScale;
	    bgScale.makeScale(osg::Vec3((bb.xMax() - bb.xMin()),1.0,(bb.zMax() - bb.zMin())));
	    _hoverBGScale->setMatrix(bgScale);
	}
	if(_hoverPoint < 0)
	{
	    _root->addChild(_hoverTransform);
	}

	_hoverGraph = selectedGraph;
	_hoverPoint = selectedPoint;
	osg::Matrix m;
	m.makeTranslate(point);
	_hoverTransform->setMatrix(m);

	return true;
    }
    else if(_hoverPoint != -1)
    {
	_root->removeChild(_hoverTransform);
	_hoverGraph = "";
	_hoverPoint = -1;
    }

    return false;
}

void DataGraph::clearHoverText()
{
    if(_hoverTransform->getNumParents())
    {
	_hoverTransform->getParent(0)->removeChild(_hoverTransform);
    }

    _hoverGraph = "";
    _hoverPoint = -1;
}

void DataGraph::makeHover()
{
    _hoverTransform = new osg::MatrixTransform();
    _hoverBGScale = new osg::MatrixTransform();
    _hoverBGGeode = new osg::Geode();
    _hoverTextGeode = new osg::Geode();

    _hoverTransform->addChild(_hoverBGScale);
    _hoverTransform->addChild(_hoverTextGeode);
    _hoverBGScale->addChild(_hoverBGGeode);

    osg::Vec4 color(0.0,0.0,0.0,1.0);

    osg::Geometry * geo = new osg::Geometry();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(osg::Vec3(1.0,-3,0));
    verts->push_back(osg::Vec3(1.0,-3,-1.0));
    verts->push_back(osg::Vec3(0,-3,-1.0));
    verts->push_back(osg::Vec3(0,-3,0));

    geo->setVertexArray(verts);

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(
	    osg::PrimitiveSet::QUADS,0);

    ele->push_back(0);
    ele->push_back(1);
    ele->push_back(2);
    ele->push_back(3);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(color);

    osg::TemplateIndexArray<unsigned int,osg::Array::UIntArrayType,4,4> *colorIndexArray;
    colorIndexArray = new osg::TemplateIndexArray<unsigned int,
		    osg::Array::UIntArrayType,4,4>;
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);

    geo->setColorArray(colors);
    geo->setColorIndices(colorIndexArray);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    _hoverBGGeode->addDrawable(geo);

    _hoverText = makeText("",osg::Vec4(1.0,1.0,1.0,1.0));
    _hoverTextGeode->addDrawable(_hoverText);
    _hoverText->setAlignment(osgText::Text::LEFT_TOP);
    osg::Vec3 pos(0,-4,0);
    _hoverText->setPosition(pos);
}

void DataGraph::update()
{
    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);
    float dataHeight = _height - (2.0 * padding);

    osg::Matrix tran,scale;
    for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
    {
	float myRangeSize;
	float myRangeCenter;

	if(it->second.xAxisType == TIMESTAMP)
	{
	    time_t range = it->second.xMaxT - it->second.xMinT;
	    time_t totalrange = _maxDisplayXT - _minDisplayXT;
	    myRangeSize = (float)((double)range / (double)totalrange);

	    time_t center = it->second.xMinT + (range / ((time_t)2.0));
	    myRangeCenter = (center - _minDisplayXT) / (double)totalrange;
	}
	else
	{
	    float range = it->second.xMax - it->second.xMin;
	    float totalrange = _maxDisplayX - _minDisplayX;
	    myRangeSize = range / totalrange;

	    float center = it->second.xMin + (range / 2.0);
	    myRangeCenter = (center - _minDisplayX) / totalrange;
	}

	float minxBound = ((0.5 * myRangeSize) - myRangeCenter) / myRangeSize;
	float maxxBound = ((0.5 * myRangeSize) + (1.0 - myRangeCenter)) / myRangeSize;

	//std::cerr << "x bounds min: " << minxBound << " max: " << maxxBound << std::endl;

	//TODO: use to adjust what points in primitives to draw

	//std::cerr << "My range size: " << myRangeSize << " range center: " << myRangeCenter << std::endl;

	osg::Matrix centerm;
	centerm.makeTranslate(osg::Vec3((myRangeCenter - 0.5) * dataWidth,0,0));
	tran.makeTranslate(osg::Vec3(-0.5,0,-0.5));
	scale.makeScale(osg::Vec3(dataWidth*myRangeSize,1.0,dataHeight));
	_graphTransformMap[it->second.name]->setMatrix(tran*scale*centerm);
    }

    tran.makeTranslate(osg::Vec3(-0.5,0,-0.5));
    scale.makeScale(osg::Vec3(_width,1.0,_height));
    _graphTransform->setMatrix(tran*scale);

    //TODO: set based on width/height
    _point->setSize(5.0);
    _lineWidth->setWidth(1.0);

    updateAxis();
    //updateClip();
}

void DataGraph::updateAxis()
{
    _axisGeode->removeDrawables(0,_axisGeode->getNumDrawables());

    if(!_dataInfoMap.size())
    {
	return;
    }

    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);
    float dataHeight = _height - (2.0 * padding);
    float dataXorigin = -(dataWidth / 2.0);
    float dataZorigin = -(dataHeight / 2.0);

    for(int i = 0; i < 2; i++)
    {
	osg::Vec3 startPoint;
	osg::Vec3 dir, tickDir;
	float totalLength;
	float startVal;

	AxisType axisType;
	float minValue, maxValue;
	time_t minTime, maxTime;
	float tickLength = _width * 0.01;	

	osgText::Text::AxisAlignment axisAlign;
	osg::Quat q;

	std::string axisLabel;
	osg::Vec4 textColor;

	// x axis
	if(i == 0)
	{
	    startPoint = osg::Vec3(dataXorigin,0,dataZorigin);
	    startVal = dataXorigin;
	    dir = osg::Vec3(1.0,0,0);
	    tickDir = osg::Vec3(0,0,1.0);
	    totalLength = dataWidth;
	    minValue = _minDisplayX;
	    maxValue = _maxDisplayX;
	    minTime = _minDisplayXT;
	    maxTime = _maxDisplayXT;
	    axisAlign = osgText::Text::XZ_PLANE;
	    textColor = osg::Vec4(0.0,0.0,0.0,1.0);
	    axisLabel = _dataInfoMap.begin()->second.xLabel;
	    axisType = _dataInfoMap.begin()->second.xAxisType;
	}
	// z axis
	else if(i == 1)
	{
	    startPoint = osg::Vec3(dataXorigin,0,dataZorigin);
	    startVal = dataZorigin;
	    dir = osg::Vec3(0,0,1.0);
	    tickDir = osg::Vec3(1.0,0,0);
	    totalLength = dataHeight;
	    axisAlign = osgText::Text::USER_DEFINED_ROTATION;
	    q.makeRotate(M_PI/2.0,osg::Vec3(1.0,0,0));
	    q = q * osg::Quat(-M_PI/2.0,osg::Vec3(0,1.0,0));

	    if(_dataInfoMap.size() == 1)
	    {
		float max = _dataInfoMap.begin()->second.zMax;
		float min = _dataInfoMap.begin()->second.zMin;
		float dataRange = max - min;
		minValue = min + (_minDisplayZ * dataRange);
		maxValue = min + (_maxDisplayZ * dataRange);
		textColor = osg::Vec4(0.0,0.0,0.0,1.0);
		axisLabel = _dataInfoMap.begin()->second.zLabel;
		axisType = _dataInfoMap.begin()->second.zAxisType;
	    }
	    else
	    {
		minValue = _minDisplayZ;
		maxValue = _maxDisplayZ;
		textColor = osg::Vec4(0.0,0.0,0.0,1.0);
		axisLabel = "Normalized Value";
		axisType = LINEAR;
	    }
	}

	osg::Geometry * geometry = new osg::Geometry();
	geometry->setUseDisplayList(false);
	geometry->setUseVertexBufferObjects(true);

	osg::Vec3Array * points = new osg::Vec3Array();
	geometry->setVertexArray(points);

	osg::Vec4 color(0.0,0.0,0.0,1.0);

	osg::Vec4Array * colorArray = new osg::Vec4Array(1);
	colorArray->at(0) = color;
	geometry->setColorArray(colorArray);
	geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

	points->push_back(startPoint);
	points->push_back(startPoint + dir * totalLength);

	switch(axisType)
	{
	    case TIMESTAMP:
	    {
		enum markInterval
		{
		    YEAR,
		    MONTH,
		    DAY,
		    HOUR,
		    MINUTE,
		    SECOND
		};

		markInterval mi;
		int intervalMult = 1;

		double totalTime = difftime(maxTime,minTime);

		if(totalTime < 20.0)
		{
		    mi = SECOND;
		}
		else
		{
		    totalTime /= 60.0;
		    if(totalTime < 20.0)
		    {
			mi = MINUTE;
		    }
		    else
		    {
			totalTime /= 60.0;
			if(totalTime < 20.0)
			{
			    mi = HOUR;
			}
			else
			{
			    totalTime /= 24.0;
			    if(totalTime < 20.0)
			    {
				mi = DAY;
			    }
			    else
			    {
				totalTime /= 30.0;
				if(totalTime < 20.0)
				{
				    mi = MONTH;
				    if(totalTime > 10.0)
				    {
					intervalMult = 2;
				    }
				}
				else
				{
				    mi = YEAR;
				    if(totalTime / 12.0 > 10)
				    {
					intervalMult = 2;
				    }
				    //std::cerr << "Setting tick value to YEAR, totalTime: " << totalTime / 12.0 << std::endl;
				}
			    }
			}
		    }
		}

		bool printYear,printMonth,printDay,printHour,printMinute;
		printYear = printMonth = printDay = printHour = printMinute = false;

		struct tm starttm, endtm;
		endtm = *gmtime(&maxTime);
		starttm = *gmtime(&minTime);

		//std::cerr << "start year: " << starttm.tm_year << " end year: " << endtm.tm_year << std::endl;

		if(starttm.tm_year != endtm.tm_year)
		{
		    printYear = printMonth = printDay = printHour = printMinute = true;
		}
		else if(starttm.tm_mon != endtm.tm_mon)
		{
		    printMonth = printDay = printHour = printMinute = true;
		}
		else if(starttm.tm_mday != endtm.tm_mday)
		{
		    printDay = printHour = printMinute = true;
		}
		else if(starttm.tm_hour != endtm.tm_hour)
		{
		    printHour = printMinute = true;
		}
		else if(starttm.tm_min != endtm.tm_min)
		{
		    printMinute = true;
		}

		struct tm currentStep;
		double currentValue;
		switch(mi)
		{
		    case YEAR:
		    {
			currentStep = *gmtime(&minTime);
			currentStep.tm_sec = currentStep.tm_min = currentStep.tm_hour = currentStep.tm_mon = 0;
			currentStep.tm_mday = 1;
			currentStep.tm_year++;

			currentValue = difftime(mktime(&currentStep),minTime);
			currentValue /= difftime(maxTime,minTime);
			currentValue *= totalLength;

			break;
		    }
		    case MONTH:
		    {
			currentStep = *gmtime(&minTime);
			currentStep.tm_sec = currentStep.tm_min = currentStep.tm_hour = 0;
			currentStep.tm_mday = 1;
			
			currentStep.tm_mon += intervalMult;
			while(currentStep.tm_mon >= 12)
			{
			    currentStep.tm_year++;
			    currentStep.tm_mon -= 12;
			}

			currentValue = difftime(mktime(&currentStep),minTime);
			currentValue /= difftime(maxTime,minTime);
			currentValue *= totalLength;
			break;
		    }
		    default:
			currentValue = totalLength + 1.0;
			std::cerr << "Unimplemented timestamp tick case." << std::endl;
			break;
		}

		while(currentValue < totalLength)
		{
		    points->push_back(startPoint + dir * currentValue);
		    points->push_back(startPoint + tickDir * tickLength + dir * currentValue);

		    std::stringstream ss;

		    switch(mi)
		    {
			case YEAR:
			    ss << currentStep.tm_year + 1900;
			    break;
			case MONTH:
			{
			    char tlabel[256];
			    strftime(tlabel,255,"%m/%y",&currentStep);
			    ss << tlabel;
			    break;
			}
			default:
			    break;
		    }

		    osgText::Text * text = makeText(ss.str(),textColor);

		    float targetSize = padding * 0.27;
		    osg::BoundingBox bb = text->getBound();
		    text->setCharacterSize(targetSize / (bb.zMax() - bb.zMin()));
		    text->setAxisAlignment(axisAlign);
		    if(axisAlign == osgText::Text::USER_DEFINED_ROTATION)
		    {
			text->setRotation(q);
		    }

		    text->setPosition(startPoint + -tickDir * (padding * 0.5 * 0.3) + dir * currentValue + osg::Vec3(0,-1,0));

		    _axisGeode->addDrawable(text);

		    switch(mi)
		    {
			case YEAR:
			    currentStep.tm_year++;
			    currentValue = difftime(mktime(&currentStep),minTime);
			    currentValue /= difftime(maxTime,minTime);
			    currentValue *= totalLength;
			    break;
			case MONTH:
			{
			    currentStep.tm_mon += intervalMult;
			    while(currentStep.tm_mon >= 12)
			    {
				currentStep.tm_year++;
				currentStep.tm_mon -= 12;
			    }

			    currentValue = difftime(mktime(&currentStep),minTime);
			    currentValue /= difftime(maxTime,minTime);
			    currentValue *= totalLength;
			    break;
			}
			default:
			    break;
		    }
		}

		break;
	    }
	    case LINEAR:
	    {
		// find tick interval
		float rangeDif = maxValue - minValue;
		int power = (int)log10(rangeDif);
		float interval = pow(10.0, power);

		while(rangeDif / interval < 2)
		{
		    interval /= 10.0;
		}

		while(rangeDif / interval > 30)
		{
		    interval *= 10.0;
		}

		if(rangeDif / interval < 4)
		{
		    interval /= 2;
		}

		float tickValue = ((float)((int)(minValue/interval)))*interval;
		if(tickValue < minValue)
		{
		    tickValue += interval;
		}
    
		float value = (((tickValue - minValue) / (maxValue - minValue)) * totalLength);
		while(value <= totalLength)
		{
		    points->push_back(startPoint + dir * value);
		    points->push_back(startPoint + tickDir * tickLength + dir * value);

		    std::stringstream ss;
		    ss << tickValue;

		    osgText::Text * text = makeText(ss.str(),textColor);

		    float targetSize = padding * 0.27;
		    osg::BoundingBox bb = text->getBound();
		    text->setCharacterSize(targetSize / (bb.zMax() - bb.zMin()));
		    text->setAxisAlignment(axisAlign);
		    if(axisAlign == osgText::Text::USER_DEFINED_ROTATION)
		    {
			text->setRotation(q);
		    }

		    text->setPosition(startPoint + -tickDir * (padding * 0.5 * 0.3) + dir * value + osg::Vec3(0,-1,0));

		    _axisGeode->addDrawable(text);

		    tickValue += interval;
		    value = (((tickValue - minValue) / (maxValue - minValue)) * totalLength);
		}

		if(!axisLabel.empty())
		{
		    osgText::Text * text = makeText(axisLabel,textColor);

		    float targetSize = padding * 0.67;
		    osg::BoundingBox bb = text->getBound();
		    text->setCharacterSize(targetSize / (bb.zMax() - bb.zMin()));
		    text->setAxisAlignment(axisAlign);
		    if(axisAlign == osgText::Text::USER_DEFINED_ROTATION)
		    {
			text->setRotation(q);
		    }

		    text->setPosition(startPoint + -tickDir * (padding * (0.3 + 0.5 * 0.7)) + dir * (totalLength / 2.0) + osg::Vec3(0,-1,0));

		    _axisGeode->addDrawable(text);
		}
		break;
	    }
	    default:
		break;
	}

	geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,points->size()));

	_axisGeode->addDrawable(geometry);
    }

    if(_dataInfoMap.size() == 1)
    {
	osgText::Text * text = makeText(_dataInfoMap.begin()->second.name,osg::Vec4(0.0,0.0,0.0,1.0));

	float targetHeight = padding * 0.95;
	float targetWidth = _width - (2.0 * padding);
	osg::BoundingBox bb = text->getBound();
	float hsize = targetHeight / (bb.zMax() - bb.zMin());
	float wsize = targetWidth / (bb.xMax() - bb.xMin());
	text->setCharacterSize(std::min(hsize,wsize));
	text->setAxisAlignment(osgText::Text::XZ_PLANE);

	text->setPosition(osg::Vec3(0,-1,(_height-padding)/2.0));

	_axisGeode->addDrawable(text);
    }
    else
    {
    }
}

void DataGraph::updateClip()
{
    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);
    float dataHeight = _height - (2.0 * padding);
    float halfWidth = dataWidth / 2.0;
    float halfHeight = dataHeight / 2.0;

    osg::Vec3 point, normal;
    osg::Plane plane;

    point = osg::Vec3(-halfWidth,0,0);
    normal = osg::Vec3(1.0,0,0);
    plane = osg::Plane(normal,point);
    _clipNode->getClipPlane(0)->setClipPlane(plane);

    point = osg::Vec3(halfWidth,0,0);
    normal = osg::Vec3(-1.0,0,0);
    plane = osg::Plane(normal,point);
    _clipNode->getClipPlane(1)->setClipPlane(plane);

    point = osg::Vec3(0,0,halfHeight);
    normal = osg::Vec3(0,0,-1.0);
    plane = osg::Plane(normal,point);
    _clipNode->getClipPlane(2)->setClipPlane(plane);

    point = osg::Vec3(0,0,-halfHeight);
    normal = osg::Vec3(0,0,1.0);
    plane = osg::Plane(normal,point);
    _clipNode->getClipPlane(3)->setClipPlane(plane);

    _clipNode->setLocalStateSetModes(); 
}

float DataGraph::calcPadding()
{
    float minD = std::min(_width,_height);

    return 0.07 * minD;
}

osgText::Text * DataGraph::makeText(std::string text, osg::Vec4 color)
{
    osgText::Text * textNode = new osgText::Text();
    textNode->setCharacterSize(1.0);
    textNode->setAlignment(osgText::Text::CENTER_CENTER);
    textNode->setColor(color);
    textNode->setBackdropColor(osg::Vec4(0,0,0,0));
    textNode->setAxisAlignment(osgText::Text::XZ_PLANE);
    textNode->setText(text);
    if(_font)
    {
	textNode->setFont(_font);
    }
    return textNode;
}
