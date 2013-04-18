#include "GroupedScatterPlot.h"
#include "ColorGenerator.h"

#include <cvrKernel/CalVR.h>
#include <cvrKernel/ComController.h>
#include <cvrConfig/ConfigManager.h>

#include <osgText/Text>

#include <iostream>
#include <sstream>
#include <cfloat>
#include <cmath>

using namespace cvr;

GroupedScatterPlot::GroupedScatterPlot(float width, float height)
{
    _width = width;
    _height = height;
    _title = "Scatter Plot";
    _firstLabel = "First";
    _secondLabel = "Second";
    _firstAxisType = GSP_LINEAR;
    _secondAxisType = GSP_LINEAR;

    _firstDataMax = FLT_MIN;
    _firstDataMin = FLT_MAX;
    _secondDataMax = FLT_MIN;
    _secondDataMin = FLT_MAX;

    _maxIndex = -1;
    _glScale = 1.0;
    _pointLineScale = ConfigManager::getFloat("value","Plugin.FuturePatient.PointLineScale",1.0);
    _masterPointScale = ConfigManager::getFloat("value","Plugin.FuturePatient.MasterPointScale",1.0);

    _root = new osg::Group();
    _bgScaleMT = new osg::MatrixTransform();
    _bgGeode = new osg::Geode();
    _pointsGeode = new osg::Geode();
    _pointsGeom = new osg::Geometry();
    _axisGeode = new osg::Geode();
    _dataGeode = new osg::Geode();

    _root->addChild(_bgScaleMT);
    _bgScaleMT->addChild(_bgGeode);
    _root->addChild(_pointsGeode);
    _pointsGeode->addDrawable(_pointsGeom);
    _root->addChild(_axisGeode);
    _root->addChild(_dataGeode);

    _font = osgText::readFontFile(CalVR::instance()->getHomeDir() + "/resources/arial.ttf");

    osg::StateSet * stateset = _root->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    _point = new osg::Point();
    stateset = _dataGeode->getOrCreateStateSet();
    stateset->setAttributeAndModes(_point,osg::StateAttribute::ON);

    _leftPaddingMult = 0.15;
    _rightPaddingMult = 0.05;
    _topPaddingMult = 0.1;
    _bottomPaddingMult = 0.1;
    _labelPaddingMult = 0.06;

    makeBG();
}

GroupedScatterPlot::~GroupedScatterPlot()
{
}

void GroupedScatterPlot::setLabels(std::string title, std::string firstLabel, std::string secondLabel)
{
    _title = title;
    _firstLabel = firstLabel;
    _secondLabel = secondLabel;

    update();
}

void GroupedScatterPlot::setAxisTypes(GSPAxisType first, GSPAxisType second)
{
    _firstAxisType = first;
    _secondAxisType = second;

    update();
}

bool GroupedScatterPlot::addGroup(int index, std::vector<std::pair<float,float> > & data)
{
    if(_plotData.find(index) != _plotData.end())
    {
	std::cerr << "Index: " << index << " has already been added to the plot." << std::endl;
	return false;
    }

    if(!data.size())
    {
	return false;
    }

    _plotData[index] = data;

    for(int i = 0; i < data.size(); ++i)
    {
	//std::cerr << "Data Point index: " << index << " first: " << data[i].first << " second: " << data[i].second << std::endl;
	if(data[i].first < _firstDataMin)
	{
	    _firstDataMin = data[i].first;
	}
	if(data[i].first > _firstDataMax)
	{
	    _firstDataMax = data[i].first;
	}
	if(data[i].second < _secondDataMin)
	{
	    _secondDataMin = data[i].second;
	}
	if(data[i].second > _secondDataMax)
	{
	    _secondDataMax = data[i].second;
	}
    }

    float lowerPadding = 0.05;

    switch(_firstAxisType)
    {
	case GSP_LINEAR:
	{
	    _firstDisplayMax = _firstDataMax;
	    _firstDisplayMin = _firstDataMin - lowerPadding * (_firstDataMax - _firstDataMin);
	    break;
	}
	case GSP_LOG:
	{
	    float logMax = log10(_firstDataMax);
	    logMax = ceil(logMax);
	    _firstDisplayMax = pow(10.0,logMax);

	    float logMin = log10(_firstDataMin);
	    logMin = logMin - lowerPadding * (logMax - logMin);
	    _firstDisplayMin = pow(10.0,logMin);
	    break;
	}
	default:
	{
	    _firstDisplayMin = _firstDataMin;
	    _firstDisplayMax = _firstDataMax;
	    break;
	}
    }

    switch(_secondAxisType)
    {
	case GSP_LINEAR:
	{
	    _secondDisplayMax = _secondDataMax;
	    _secondDisplayMin = _secondDataMin - lowerPadding * (_secondDataMax - _secondDataMin);
	    break;
	}
	case GSP_LOG:
	{
	    float logMax = log10(_secondDataMax);
	    logMax = ceil(logMax);
	    _secondDisplayMax = pow(10.0,logMax);

	    float logMin = log10(_secondDataMin);
	    logMin = logMin - lowerPadding * (logMax - logMin);
	    _secondDisplayMin = pow(10.0,logMin);
	    break;
	}
	default:
	{
	    _secondDisplayMin = _secondDataMin;
	    _secondDisplayMax = _secondDataMax;
	    break;
	}
    }

    if(index > _maxIndex)
    {
	_maxIndex = index;
    }

    update();

    return true;
}

void GroupedScatterPlot::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;

    update();
}

void GroupedScatterPlot::setGLScale(float scale)
{
    _glScale = scale;
    update();
}

void GroupedScatterPlot::makeBG()
{
    osg::Geometry * geom = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array(8);
    verts->at(0) = osg::Vec3(0.5,1,0.5);
    verts->at(1) = osg::Vec3(0.5,1,-0.5);
    verts->at(3) = osg::Vec3(-0.5,1,0.5);
    verts->at(2) = osg::Vec3(-0.5,1,-0.5);
    verts->at(4) = osg::Vec3(0.5-_rightPaddingMult,0.5,0.5-_topPaddingMult);
    verts->at(5) = osg::Vec3(0.5-_rightPaddingMult,0.5,-0.5+_bottomPaddingMult);
    verts->at(7) = osg::Vec3(-0.5+_leftPaddingMult,0.5,0.5-_topPaddingMult);
    verts->at(6) = osg::Vec3(-0.5+_leftPaddingMult,0.5,-0.5+_bottomPaddingMult);

    osg::Vec4 bgColor(0.9,0.9,0.9,1.0);
    osg::Vec4 dataBGColor(0.4,0.4,0.4,1.0);

    osg::Vec4Array * colors = new osg::Vec4Array(8);
    colors->at(0) = bgColor;
    colors->at(1) = bgColor;
    colors->at(2) = bgColor;
    colors->at(3) = bgColor;
    colors->at(4) = dataBGColor;
    colors->at(5) = dataBGColor;
    colors->at(6) = dataBGColor;
    colors->at(7) = dataBGColor;

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setVertexArray(verts);
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,8));
    geom->getBound();

    _bgGeode->addDrawable(geom);
}


void GroupedScatterPlot::update()
{
    updateSizes();

    osg::Vec3 scale(_width,1.0,_height);
    osg::Matrix scaleMat;
    scaleMat.makeScale(scale);
    _bgScaleMT->setMatrix(scaleMat);

    updateAxis();
    updateGraph();

    float avglen = (_width + _height) / 2.0;
    _point->setSize(_glScale * avglen * 0.04 * _pointLineScale);

    if(ComController::instance()->isMaster())
    {
	_point->setSize(_point->getSize() * _masterPointScale);
    }
}

void GroupedScatterPlot::updateAxis()
{
    _axisGeode->removeDrawables(0,_axisGeode->getNumDrawables());
    if(!_plotData.size())
    {
	return;
    }

    //title text
    {
	osgText::Text * text = makeText(_title, osg::Vec4(0,0,0,1));
	float csize1,csize2;

	osg::BoundingBox bb = text->getBound();
	csize1 = (0.90*_width) / (bb.xMax() - bb.xMin());
	csize2 = (0.90*_height*_topPaddingMult) / (bb.zMax() - bb.zMin());
	text->setCharacterSize(std::min(csize1,csize2));
	text->setPosition(osg::Vec3(0,0,(_height/2.0)-(0.5*_topPaddingMult*_height)));

	//std::cerr << "Made title text: " << _title << " csize1: " << csize1 << " csize2: " << csize2 << std::endl;

	_axisGeode->addDrawable(text);
    }

    osg::Quat q;
    q.makeRotate(M_PI/2.0,osg::Vec3(1.0,0,0));
    q = q * osg::Quat(-M_PI/2.0,osg::Vec3(0,1.0,0));

    float labelTextSize, labelLeftSize, labelBottomSize;
    labelTextSize = std::min(_width,_height)*_labelPaddingMult;
    labelLeftSize = (_leftPaddingMult * _width) - labelTextSize;
    labelBottomSize = (_bottomPaddingMult * _height) - labelTextSize;

    //axis labels
    {
	float csize1,csize2;

	osgText::Text * text = makeText(_firstLabel,osg::Vec4(0,0,0,1));
	
	osg::BoundingBox bb = text->getBound();
	csize1 = (0.80*_width) / (bb.xMax() - bb.xMin());
	csize2 = (0.9*labelTextSize) / (bb.zMax() - bb.zMin());
	text->setCharacterSize(std::min(csize1,csize2));
	text->setPosition(osg::Vec3(0,0,(-_height/2.0)+(0.5*labelTextSize)));

	_axisGeode->addDrawable(text);

	text = makeText(_secondLabel,osg::Vec4(0,0,0,1));
	text->setRotation(q);

	bb = text->getBound();
	csize1 = (0.8*_height) / (bb.zMax() - bb.zMin());
	csize2 = (0.9*labelTextSize) / (bb.xMax() - bb.xMin());
	text->setCharacterSize(std::min(csize1,csize2));
	text->setPosition(osg::Vec3((-_width/2.0)+(0.5*labelTextSize),0,0));

	_axisGeode->addDrawable(text);
    }

    osg::Geometry * lineGeom = new osg::Geometry();
    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(0,0,0,1);

    lineGeom->setColorArray(colors);
    lineGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    lineGeom->setVertexArray(verts);
    lineGeom->setUseDisplayList(false);
    lineGeom->setUseVertexBufferObjects(true);

    //border
    verts->push_back(osg::Vec3(_graphLeft,-1,_graphTop));
    verts->push_back(osg::Vec3(_graphLeft,-1,_graphBottom));
    verts->push_back(osg::Vec3(_graphLeft,-1,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,-1,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,-1,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,-1,_graphTop));
    verts->push_back(osg::Vec3(_graphRight,-1,_graphTop));
    verts->push_back(osg::Vec3(_graphLeft,-1,_graphTop));

    //tick marks/labels
    float tickSize = std::min(_width,_height) * 0.01;
    switch(_firstAxisType)
    {
	case GSP_LINEAR:
	{
	    std::cerr << "Linear graph label not yet implemented." << std::endl;
	    break;
	}
	case GSP_LOG:
	{
	    float tickLoc = _graphRight;
	    float currentTickValue = _firstDisplayMax;
	    float interval = (1.0 / (log10(_firstDisplayMax) - log10(_firstDisplayMin))) * (_graphRight - _graphLeft);

	    float tickCharacterSize;
	    int maxExp = (int)std::max(fabs(log10(_firstDisplayMax)),fabs(log10(_firstDisplayMin)));
	    maxExp += 2;

	    {
		std::stringstream testss;
		while(maxExp > 0)
		{
		    testss << "0";
		    maxExp--;
		}

		osg::ref_ptr<osgText::Text> testText = makeText(testss.str(),osg::Vec4(0,0,0,1));
		osg::BoundingBox testbb = testText->getBound();
		float testHeight = testbb.zMax() - testbb.zMin();

		tickCharacterSize = (labelBottomSize * 0.95 - 2.0 * tickSize) / testHeight;
	    }

	    while(tickLoc >= _graphLeft)
	    {
		verts->push_back(osg::Vec3(tickLoc,-1,_graphBottom));
		verts->push_back(osg::Vec3(tickLoc,-1,_graphBottom-tickSize));

		std::stringstream tss;
		tss << currentTickValue;
		osgText::Text * tickText = makeText(tss.str(),osg::Vec4(0,0,0,1));
		tickText->setAlignment(osgText::Text::CENTER_TOP);
		tickText->setCharacterSize(tickCharacterSize);
		tickText->setPosition(osg::Vec3(tickLoc,-1,_graphBottom - 2.0*tickSize));
		_axisGeode->addDrawable(tickText);

		currentTickValue /= 10.0;
		tickLoc -= interval;
	    }

	    tickLoc = _graphRight;
	    currentTickValue = _firstDisplayMax;

	    int count = -1;
	    float tickReduc = _firstDisplayMax / 10.0;
	    while(tickLoc >= _graphLeft)
	    { 
		count++;
		verts->push_back(osg::Vec3(tickLoc,-1,_graphBottom));
		verts->push_back(osg::Vec3(tickLoc,-1,_graphBottom - 0.5*tickSize));

		if((count % 10) == 9)
		{
		    tickReduc /= 10.0;
		    count++;
		}
		currentTickValue -= tickReduc;
		tickLoc = ((log10(currentTickValue) - log10(_firstDisplayMin)) / (log10(_firstDisplayMax) - log10(_firstDisplayMin))) * (_graphRight - _graphLeft);
		tickLoc += _graphLeft;
	    }

	    break;
	}
	default: 
	{
	    std::cerr << "Unknown axis type." << std::endl;
	    break;
	}
    }

    switch(_secondAxisType)
    {
	case GSP_LINEAR:
	{
	    std::cerr << "Linear graph label not yet implemented." << std::endl;
	    break;
	}
	case GSP_LOG:
	{
	    float tickLoc = _graphTop;
	    float currentTickValue = _secondDisplayMax;
	    float interval = (1.0 / (log10(_secondDisplayMax) - log10(_secondDisplayMin))) * (_graphTop - _graphBottom);

	    float tickCharacterSize;
	    int maxExp = (int)std::max(fabs(log10(_secondDisplayMax)),fabs(log10(_secondDisplayMin)));
	    maxExp += 2;

	    {
		std::stringstream testss;
		while(maxExp > 0)
		{
		    testss << "0";
		    maxExp--;
		}

		osg::ref_ptr<osgText::Text> testText = makeText(testss.str(),osg::Vec4(0,0,0,1));
		osg::BoundingBox testbb = testText->getBound();
		float testWidth = testbb.xMax() - testbb.xMin();

		tickCharacterSize = (labelLeftSize * 0.95 - 2.0 * tickSize) / testWidth;
	    }

	    while(tickLoc >= _graphBottom)
	    {
		verts->push_back(osg::Vec3(_graphLeft,-1,tickLoc));
		verts->push_back(osg::Vec3(_graphLeft-tickSize,-1,tickLoc));

		std::stringstream tss;
		tss << currentTickValue;
		osgText::Text * tickText = makeText(tss.str(),osg::Vec4(0,0,0,1));
		tickText->setAlignment(osgText::Text::RIGHT_CENTER);
		tickText->setCharacterSize(tickCharacterSize);
		tickText->setPosition(osg::Vec3(_graphLeft - 2.0*tickSize,-1,tickLoc));
		_axisGeode->addDrawable(tickText);

		currentTickValue /= 10.0;
		tickLoc -= interval;
	    }

	    tickLoc = _graphTop;
	    currentTickValue = _secondDisplayMax;

	    int count = -1;
	    float tickReduc = _secondDisplayMax / 10.0;
	    while(tickLoc >= _graphBottom)
	    { 
		count++;
		verts->push_back(osg::Vec3(_graphLeft,-1,tickLoc));
		verts->push_back(osg::Vec3(_graphLeft- 0.5*tickSize,-1,tickLoc));

		if((count % 10) == 9)
		{
		    tickReduc /= 10.0;
		    count++;
		}
		currentTickValue -= tickReduc;
		tickLoc = ((log10(currentTickValue) - log10(_secondDisplayMin)) / (log10(_secondDisplayMax) - log10(_secondDisplayMin))) * (_graphTop - _graphBottom);
		tickLoc += _graphBottom;
	    }

	    break;
	}
	default: 
	{
	    std::cerr << "Unknown axis type." << std::endl;
	    break;
	}
    }

    lineGeom->addPrimitiveSet(new osg::DrawArrays(GL_LINES,0,verts->size()));
    _axisGeode->addDrawable(lineGeom);
    lineGeom->getBound();
}

void GroupedScatterPlot::updateGraph()
{
    if(!_plotData.size())
    {
	return;
    }

    _dataGeode->removeDrawables(0,_dataGeode->getNumDrawables());

    osg::Geometry * geom = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colors = new osg::Vec4Array();

    geom->setVertexArray(verts);
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);

    float firstLogMin = log10(_firstDisplayMin);
    float firstLogMax = log10(_firstDisplayMax);
    float secondLogMin = log10(_secondDisplayMin);
    float secondLogMax = log10(_secondDisplayMax);
    for(std::map<int,std::vector<std::pair<float,float> > >::iterator it = _plotData.begin(); it != _plotData.end(); ++it)
    {
	osg::Vec4 color = ColorGenerator::makeColor(it->first,_maxIndex+1);

	for(int i = 0; i < it->second.size(); ++i)
	{
	    bool addPoint = true;
	    float pointX, pointZ;

	    switch(_firstAxisType)
	    {
		case GSP_LINEAR:
		    {
			addPoint = false;
			break;
		    }
		case GSP_LOG:
		    {
			float logVal = log10(it->second[i].first);
			if(logVal < firstLogMin || logVal > firstLogMax)
			{
			    addPoint = false;
			}
			else
			{
			    pointX = _graphLeft + ((logVal - firstLogMin) / (firstLogMax - firstLogMin)) * (_graphRight - _graphLeft);
			}
			break;
		    }
		default:
		    {
			addPoint = false;
			break;
		    }
	    }

	    switch(_secondAxisType)
	    {
		case GSP_LINEAR:
		    {
			addPoint = false;
			break;
		    }
		case GSP_LOG:
		    {
			float logVal = log10(it->second[i].second);
			if(logVal < secondLogMin || logVal > secondLogMax)
			{
			    addPoint = false;
			}
			else
			{
			    pointZ = _graphBottom + ((logVal - secondLogMin) / (secondLogMax - secondLogMin)) * (_graphTop - _graphBottom);
			}
			break;
		    }
		default:
		    {
			addPoint = false;
			break;
		    }
	    }

	    if(addPoint)
	    {
		verts->push_back(osg::Vec3(pointX,0,pointZ));
		colors->push_back(color);
	    }
	}
    }


    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,verts->size()));
    geom->getBound();

    _dataGeode->addDrawable(geom);
}

void GroupedScatterPlot::updateSizes()
{
    _graphLeft = -(_width / 2.0) + (_leftPaddingMult * _width);
    _graphRight = (_width / 2.0) - (_rightPaddingMult * _width);
    _graphTop = (_height / 2.0) - (_topPaddingMult * _height);
    _graphBottom = -(_height / 2.0) + (_bottomPaddingMult * _height);
}

osgText::Text * GroupedScatterPlot::makeText(std::string text, osg::Vec4 color)
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
