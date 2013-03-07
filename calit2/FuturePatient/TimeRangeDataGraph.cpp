#include "TimeRangeDataGraph.h"
#include "ColorGenerator.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/CalVR.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/OsgMath.h>

#include <iostream>
#include <sstream>
#include <ctime>

using namespace cvr;

TimeRangeDataGraph::TimeRangeDataGraph()
{
    _root = new osg::Group();
    _root->setCullingActive(false);
    _bgScaleMT = new osg::MatrixTransform();
    _axisGeode = new osg::Geode();
    _bgGeode = new osg::Geode();
    _graphGeode = new osg::Geode();
    _graphGeode->setCullingActive(false);

    _shadingGeode = new osg::Geode();
    _shadingGeode->setCullingActive(false);

    _root->addChild(_bgScaleMT);
    _root->addChild(_axisGeode);
    _root->addChild(_graphGeode);
    _root->addChild(_shadingGeode);
    _bgScaleMT->addChild(_bgGeode);

    _width = _height = 1000.0;

    //TODO: add checks in higher level to ignore these values if not set
    _timeMin = _displayMin = 0;
    _timeMax = _displayMax = 0;

    osg::StateSet * stateset = _root->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    _font = osgText::readFontFile(CalVR::instance()->getHomeDir() + "/resources/arial.ttf");

    _currentHoverIndex = -1;
    _currentHoverGraph = -1;

    _barPos = 0;
    _glScale = 1.0;
    _pointLineScale = ConfigManager::getFloat("value","Plugin.FuturePatient.PointLineScale",1.0);
    _masterLineScale = ConfigManager::getFloat("value","Plugin.FuturePatient.MasterLineScale",1.0);

    makeBG();
    makeHover();
    makeBar();

    _graphGeode->getOrCreateStateSet()->setAttributeAndModes(_barLineWidth,osg::StateAttribute::ON);
    _graphGeode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    _graphGeode->getOrCreateStateSet()->setMode(GL_BLEND,osg::StateAttribute::ON);

    update();
}

TimeRangeDataGraph::~TimeRangeDataGraph()
{
}

void TimeRangeDataGraph::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;

    update();
}

bool TimeRangeDataGraph::getBarVisible()
{
    return _barTransform->getNumParents();
}

void TimeRangeDataGraph::setBarVisible(bool b)
{
    if(b == getBarVisible())
    {
	return;
    }

    if(b)
    {
	_root->addChild(_barTransform);
    }
    else
    {
	_root->removeChild(_barTransform);
    }
}

float TimeRangeDataGraph::getBarPosition()
{
    return _barPos;
}

void TimeRangeDataGraph::setBarPosition(float pos)
{
    if(!_barPosTransform)
    {
	return;
    }

    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);

    osg::Matrix trans;
    trans.makeTranslate(osg::Vec3((dataWidth*pos)-(dataWidth/2.0),0,0));
    _barPosTransform->setMatrix(trans);

    _barPos = pos;
}

bool TimeRangeDataGraph::getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point)
{
    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);
    float dataHeight = _height - (2.0 * padding);

    osg::Vec3 point1,point2(0,1000.0,0),planePoint,planeNormal(0,-1,0),intersect;
    float w;
    point1 = point1 * mat;
    point2 = point2 * mat;

    if(linePlaneIntersectionRef(point1,point2,planePoint,planeNormal,intersect,w))
    {
	if(fabs(intersect.x()) > (_width/2.0) || fabs(intersect.z()) > (_height / 2.0))
	{
	    return false;
	}
	intersect.x() /= dataWidth;
	intersect.z() /= dataHeight;
	intersect.x() += 0.5;
	intersect.z() += 0.5;
	point = intersect;
    }
    else
    {
	return false;
    }

    return true;
}

void TimeRangeDataGraph::setGLScale(float scale)
{
    _glScale = scale;
    update();
}

void TimeRangeDataGraph::addGraph(std::string name, std::vector<std::pair<time_t,time_t> > & rangeList, std::vector<int> & valueList, int maxValue)
{
    if(!rangeList.size())
    {
	return;
    }

    for(int i = 0; i < _graphList.size(); ++i)
    {
	if(_graphList[i]->name == name)
	{
	    return;
	}
    }

    time_t min;
    time_t max;
    min = rangeList[0].first;
    max = rangeList[0].second;

    for(int i = 1; i < rangeList.size(); ++i)
    {
	if(rangeList[i].first < min)
	{
	    min = rangeList[i].first;
	}
	if(rangeList[i].second > max)
	{
	    max = rangeList[i].second;
	}
    }

    if(_timeMin == 0 || min < _timeMin)
    {
	_timeMin = min;
    }

    if(_timeMax == 0 || max > _timeMax)
    {
	_timeMax = max;
    }

    _displayMin = _timeMin;
    _displayMax = _timeMax;

    RangeDataInfo * rdi = new RangeDataInfo;
    rdi->name = name;
    rdi->ranges = rangeList;
    rdi->values = valueList;
    rdi->maxValue = maxValue;
    rdi->barGeometry = new osg::Geometry();
    rdi->barOutlineGeometry = new osg::Geometry();
    _graphGeode->addDrawable(rdi->barGeometry);
    _graphGeode->addDrawable(rdi->barOutlineGeometry);

    _graphList.push_back(rdi);

    initGeometry(rdi);

    update();
}

void TimeRangeDataGraph::setDisplayRange(time_t & start, time_t & end)
{
    _displayMin = start;
    _displayMax = end;
    update();
}

void TimeRangeDataGraph::getDisplayRange(time_t & start, time_t & end)
{
    start = _displayMin;
    end = _displayMax;
}

time_t TimeRangeDataGraph::getMaxTimestamp()
{
    return _timeMax;
}

time_t TimeRangeDataGraph::getMinTimestamp()
{
    return _timeMin;
}

osg::Group * TimeRangeDataGraph::getGraphRoot()
{
    return _root.get();
}

void TimeRangeDataGraph::setHover(osg::Vec3 intersect)
{
    if(!_hoverGeode)
    {
	return;
    }

    int graph = -1;
    int index = -1;

    float myTop = _graphTop;
    for(int i = 0; i < _graphList.size(); ++i)
    {
	if(intersect.z() <= myTop && intersect.z() >= myTop - _barHeight)
	{
	    if(intersect.x() < _graphLeft || intersect.x() > _graphRight)
	    {
		break;
	    }

	    time_t intersectTime =  _displayMin + ((time_t)(((intersect.x() - _graphLeft) / (_graphRight - _graphLeft)) * ((double)(_displayMax-_displayMin))));
	    for(int j = 0; j < _graphList[i]->ranges.size(); ++j)
	    {
		if(intersectTime >= _graphList[i]->ranges[j].first && intersectTime <= _graphList[i]->ranges[j].second)
		{
		    if(_graphList[i]->values[j] != 0)
		    {
			graph = i;
			index = j;
		    }
		    break;
		}
	    }

	    break;
	}

	myTop -= _barHeight + _barPadding;
    }

    if(graph < 0 || index < 0)
    {
	clearHoverText();
    }
    else if(graph != _currentHoverGraph || index != _currentHoverIndex)
    {
	std::stringstream ss;
	ss << _graphList[graph]->name << std::endl;
	ss << "Start: " << ctime(&_graphList[graph]->ranges[index].first);
	ss << "End: " << ctime(&_graphList[graph]->ranges[index].second);
	if(_labelMap.find(_graphList[graph]->values[index]) != _labelMap.end())
	{
	    ss << "Value: " << _labelMap[_graphList[graph]->values[index]];
	}
	else
	{
	    ss << "Value: " << _graphList[graph]->values[index];
	}

	_hoverText->setCharacterSize(1.0);
	_hoverText->setText(ss.str());
	_hoverText->setAlignment(osgText::Text::LEFT_TOP);
	osg::BoundingBox bb = _hoverText->getBound();
	float csize = 150.0 / (bb.zMax() - bb.zMin());
	_hoverText->setCharacterSize(csize);
	_hoverText->setPosition(osg::Vec3(intersect.x(),-2.5,intersect.z()));

	float bgheight = (bb.zMax() - bb.zMin()) * csize;
	float bgwidth = (bb.xMax() - bb.xMin()) * csize;
	osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_hoverBGGeom->getVertexArray());
	if(verts)
	{
	    verts->at(0) = osg::Vec3(intersect.x()+bgwidth,-2,intersect.z()-bgheight);
	    verts->at(1) = osg::Vec3(intersect.x()+bgwidth,-2,intersect.z());
	    verts->at(2) = osg::Vec3(intersect.x(),-2,intersect.z());
	    verts->at(3) = osg::Vec3(intersect.x(),-2,intersect.z()-bgheight);
	    verts->dirty();
	}

	_currentHoverGraph = graph;
	_currentHoverIndex = index;

	if(!_hoverGeode->getNumParents())
	{
	    _root->addChild(_hoverGeode);
	}
    }
}

void TimeRangeDataGraph::clearHoverText()
{
    if(!_hoverGeode)
    {
	return;
    }

    _currentHoverIndex = -1;
    _currentHoverGraph = -1;

    if(_hoverGeode->getNumParents())
    {
	_root->removeChild(_hoverGeode);
    }
}

float TimeRangeDataGraph::calcPadding()
{
    float minD = std::min(_width,_height);

    return 0.07 * minD;
}

void TimeRangeDataGraph::initGeometry(RangeDataInfo * rdi)
{
    int validCount = 0;
    for(int i = 0; i < rdi->values.size(); ++i)
    {
	if(rdi->values[i] != 0)
	{
	    validCount++;
	}
    }

    rdi->barOutlineGeometry->setUseDisplayList(false);
    rdi->barOutlineGeometry->setUseVertexBufferObjects(true);

    osg::Vec3Array * verts = new osg::Vec3Array(4*rdi->ranges.size());
    osg::Vec4Array * colors = new osg::Vec4Array(1);

    colors->at(0) = osg::Vec4(1.0,1.0,1.0,1.0);

    rdi->barOutlineGeometry->setVertexArray(verts);
    rdi->barOutlineGeometry->setColorArray(colors);
    rdi->barOutlineGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    rdi->barOutlineGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,4*rdi->ranges.size()));

    rdi->barGeometry->setUseDisplayList(false);
    rdi->barGeometry->setUseVertexBufferObjects(true);

    verts = new osg::Vec3Array(4*validCount);
    colors = new osg::Vec4Array(4*validCount);

    rdi->barGeometry->setVertexArray(verts);
    rdi->barGeometry->setColorArray(colors);
    rdi->barGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    rdi->barGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4*validCount));

    int colorIndex = 0;
    for(int i = 0; i < rdi->values.size(); ++i)
    {
	if(rdi->values[i] == 0)
	{
	    continue;
	}
    
	float alpha = rdi->values[i] / ((float)rdi->maxValue);
	colors->at(colorIndex).w() = alpha;
	colors->at(colorIndex+1).w() = alpha;
	colors->at(colorIndex+2).w() = alpha;
	colors->at(colorIndex+3).w() = alpha;
	
	colorIndex += 4;
    }
    colors->dirty();
}

void TimeRangeDataGraph::makeBG()
{
    osg::Geometry * geom = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array(4);
    verts->at(0) = osg::Vec3(0.5,1,0.5);
    verts->at(1) = osg::Vec3(0.5,1,-0.5);
    verts->at(2) = osg::Vec3(-0.5,1,0.5);
    verts->at(3) = osg::Vec3(-0.5,1,-0.5);

    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(1.0,1.0,1.0,1.0);

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->setVertexArray(verts);
    geom->setUseDisplayList(false);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP,0,4));

    _bgGeode->addDrawable(geom);
}

void TimeRangeDataGraph::makeHover()
{
    _hoverGeode = new osg::Geode();
    _hoverBGGeom = new osg::Geometry();
    _hoverBGGeom->setUseDisplayList(false);
    _hoverText = makeText("",osg::Vec4(1,1,1,1));
    _hoverGeode->addDrawable(_hoverBGGeom);
    _hoverGeode->addDrawable(_hoverText);
    _hoverGeode->setCullingActive(false);

    osg::Vec3Array * verts = new osg::Vec3Array(4);
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(0,0,0,1);

    _hoverBGGeom->setVertexArray(verts);
    _hoverBGGeom->setColorArray(colors);
    _hoverBGGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    _hoverBGGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));
}

void TimeRangeDataGraph::makeBar()
{
    _barTransform = new osg::MatrixTransform();
    _barPosTransform = new osg::MatrixTransform();
    _barGeode = new osg::Geode();
    _barGeometry = new osg::Geometry();
    _barTransform->addChild(_barPosTransform);
    _barPosTransform->addChild(_barGeode);
    _barGeode->addDrawable(_barGeometry);

    osg::Geometry * geo = _barGeometry.get();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(osg::Vec3(0,-0.2,-0.5));
    verts->push_back(osg::Vec3(0,-0.2,0.5));
    geo->setVertexArray(verts);

    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(1,1,0,1);
    geo->setColorArray(colors);
    geo->setColorBinding(osg::Geometry::BIND_OVERALL);

    geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,2));

    _barLineWidth = new osg::LineWidth();
    _barGeode->getOrCreateStateSet()->setAttributeAndModes(_barLineWidth,osg::StateAttribute::ON);
    _barGeode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    _barGeode->getOrCreateStateSet()->setMode(GL_BLEND,osg::StateAttribute::ON);
}

void TimeRangeDataGraph::update()
{
    updateSizes();

    float barHeight = _graphTop - _graphBottom;

    osg::Vec3 scale(_width,1.0,_height);
    osg::Matrix scaleMat;
    scaleMat.makeScale(scale);
    _bgScaleMT->setMatrix(scaleMat);

    updateGraphs();
    updateAxis();
    updateShading();

    scaleMat.makeScale(osg::Vec3(1,1,barHeight));
    _barTransform->setMatrix(scaleMat);

    setBarPosition(_barPos);

    float avglen = (_width + _height) / 2.0;
    _barLineWidth->setWidth(_glScale * avglen * 0.05 * _pointLineScale * _pointLineScale);

    if(ComController::instance()->isMaster())
    {
	_barLineWidth->setWidth(_barLineWidth->getWidth() * _masterLineScale);
    }
}

void TimeRangeDataGraph::updateGraphs()
{
    if(!_graphList.size())
    {
	return;
    }

    //update bar colors
    for(int i = 0; i < _graphList.size(); ++i)
    {
	osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_graphList[i]->barOutlineGeometry->getColorArray());
	osg::Vec4 myColor = ColorGenerator::makeColor(i,_graphList.size());
	if(colors)
	{
	    colors->at(0) = myColor;
	    colors->dirty();
	}

	colors = dynamic_cast<osg::Vec4Array*>(_graphList[i]->barGeometry->getColorArray());
	if(colors)
	{
	    for(int j = 0; j < colors->size(); ++j)
	    {
		colors->at(j).x() = myColor.x();
		colors->at(j).y() = myColor.y();
		colors->at(j).z() = myColor.z();
	    }
	    colors->dirty();
	}
    }

    //update bar points
    float myTop = _graphTop;
    for(int i = 0; i < _graphList.size(); ++i)
    {
	osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_graphList[i]->barGeometry->getVertexArray());
	osg::Vec3Array * outlineVerts = dynamic_cast<osg::Vec3Array*>(_graphList[i]->barOutlineGeometry->getVertexArray());

	if(verts && outlineVerts)
	{
	    int index = 0;
	    int validIndex = 0;
	    for(int j = 0; j < _graphList[i]->ranges.size(); ++j)
	    {
		time_t start = _graphList[i]->ranges[j].first;
		time_t end = _graphList[i]->ranges[j].second;
		if(start < _displayMin)
		{
		    start = _displayMin;
		}
		else if(start > _displayMax)
		{
		    start = _displayMax;
		}

		if(end < _displayMin)
		{
		    end = _displayMin;
		}
		else if(end > _displayMax)
		{
		    end = _displayMax;
		}

		if(start != end)
		{
		    float barLeft = (((double)(start - _displayMin)) / ((double)(_displayMax-_displayMin))) * (_graphRight-_graphLeft) + _graphLeft;
		    float barRight = (((double)(end - _displayMin)) / ((double)(_displayMax-_displayMin))) * (_graphRight-_graphLeft) + _graphLeft;
		    outlineVerts->at(index) = osg::Vec3(barRight,-0.5,myTop);
		    outlineVerts->at(index+1) = osg::Vec3(barLeft,-0.5,myTop);
		    outlineVerts->at(index+2) = osg::Vec3(barLeft,-0.5,myTop-_barHeight);
		    outlineVerts->at(index+3) = osg::Vec3(barRight,-0.5,myTop-_barHeight);

		    if(_graphList[i]->values[j] != 0)
		    {
			verts->at(validIndex) = osg::Vec3(barRight,0,myTop);
			verts->at(validIndex+1) = osg::Vec3(barLeft,0,myTop);
			verts->at(validIndex+2) = osg::Vec3(barLeft,0,myTop-_barHeight);
			verts->at(validIndex+3) = osg::Vec3(barRight,0,myTop-_barHeight);
			validIndex += 4;
		    }
		}
		else
		{
		    outlineVerts->at(index) = osg::Vec3(0,0,0);
		    outlineVerts->at(index+1) = osg::Vec3(0,0,0);
		    outlineVerts->at(index+2) = osg::Vec3(0,0,0);
		    outlineVerts->at(index+3) = osg::Vec3(0,0,0);

		    if(_graphList[i]->values[j] != 0)
		    {
			verts->at(validIndex) = osg::Vec3(0,0,0);
			verts->at(validIndex+1) = osg::Vec3(0,0,0);
			verts->at(validIndex+2) = osg::Vec3(0,0,0);
			verts->at(validIndex+3) = osg::Vec3(0,0,0);
			validIndex += 4;
		    }
		}

		index += 4;
	    }
	    verts->dirty();
	    outlineVerts->dirty();
	}

	myTop -= _barHeight + _barPadding;
    }
}

void TimeRangeDataGraph::updateAxis()
{
    _axisGeode->removeDrawables(0,_axisGeode->getNumDrawables());

    if(!_graphList.size())
    {
	return;
    }

    osg::Geometry * lineGeom = new osg::Geometry();
    osg::Vec3Array * lineVerts = new osg::Vec3Array();
    osg::Vec4Array * lineColors = new osg::Vec4Array(1);

    lineColors->at(0) = osg::Vec4(0,0,0,1);

    lineGeom->setUseDisplayList(false);
    lineGeom->setUseVertexBufferObjects(true);
    lineGeom->setVertexArray(lineVerts);
    lineGeom->setColorArray(lineColors);
    lineGeom->setColorBinding(osg::Geometry::BIND_OVERALL);

    lineVerts->push_back(osg::Vec3(_graphLeft,0,_graphTop));
    lineVerts->push_back(osg::Vec3(_graphLeft,0,_graphBottom));

    osg::Quat q;
    q.makeRotate(M_PI/2.0,osg::Vec3(1.0,0,0));
    q = q * osg::Quat(-M_PI/2.0,osg::Vec3(0,1.0,0));

    float padding = calcPadding();

    float myCenter = _graphTop - (_barHeight / 2.0);
    for(int i = 0; i < _graphList.size(); ++i)
    {
	osgText::Text * text = makeText(_graphList[i]->name,osg::Vec4(0,0,0,1));
	text->setRotation(q);
	text->setAlignment(osgText::Text::CENTER_CENTER);
	osg::BoundingBox bb = text->getBound();
	float csize1 = (padding * 0.8) / (bb.xMax() - bb.xMin());
	float csize2 = _barHeight / (bb.zMax() - bb.zMin());
	text->setCharacterSize(std::min(csize1,csize2));
	text->setPosition(osg::Vec3(_graphLeft - (padding / 2.0),0,myCenter));
	_axisGeode->addDrawable(text);

	myCenter -= _barHeight + _barPadding;
    }

    lineGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,lineVerts->size()));
    _axisGeode->addDrawable(lineGeom);
}

void TimeRangeDataGraph::updateShading()
{
    _shadingGeode->removeDrawables(0,_shadingGeode->getNumDrawables());

    osg::Geometry * geom = new osg::Geometry();
    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colors = new osg::Vec4Array();
    geom->setVertexArray(verts);
    geom->setColorArray(colors);
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);

    osg::Vec4 defaultColor(0.4,0.4,0.4,1.0);
    verts->push_back(osg::Vec3(_graphLeft,0.6,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,0.6,_graphBottom));
    verts->push_back(osg::Vec3(_graphRight,0.6,_graphTop));
    verts->push_back(osg::Vec3(_graphLeft,0.6,_graphTop));
    colors->push_back(defaultColor);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));

    _shadingGeode->addDrawable(geom);
}

void TimeRangeDataGraph::updateSizes()
{
    float padding = calcPadding();

    _graphLeft = -_width / 2.0 + padding;
    _graphRight = _width / 2.0 - padding;
    _graphTop = _height / 2.0 - padding;
    _graphBottom = -_height / 2.0 + padding;

    _barHeight = (_graphTop - _graphBottom) / ((float)_graphList.size());
    _barPadding = 0.0;

    if(_graphList.size() > 1)
    {
	_barPadding = (0.05 * (_graphTop - _graphBottom)) / ((float)(_graphList.size()-1));
	_barHeight *= 0.95;
    }
}

osgText::Text * TimeRangeDataGraph::makeText(std::string text, osg::Vec4 color)
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
